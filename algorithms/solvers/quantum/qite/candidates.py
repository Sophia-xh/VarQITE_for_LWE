# -*- coding: utf-8 -*-
"""
VarQITE candidate generation and LWE-prior reranking utilities.

This module turns the decoded samples produced by the local VarQITE solver into
multiple local correction candidates, evaluates each candidate, and selects a
candidate-best solution using rules that do not use the true secret.

The formal algorithmic selection should use only ``best_candidate``. Fields such
as ``any_secret_success`` and ``secret_success`` are diagnostic fields for
experiments when the true LWE instance is known.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from algorithms.recovery.recover_secret import recover_secret_from_lattice_point

CandidateSelectionMode = Literal[
    "energy_then_count",
    "count_then_energy",
    "energy",
    "count",
    "sparse_energy_then_count",
    "lwe_prior",
    "energy_lwe_prior",
]


@dataclass
class CandidateRerankConfig:
    """Configuration for VarQITE candidate generation and reranking.

    Parameters
    ----------
    candidate_topk:
        Number of decoded candidates to keep from ``decoded_count_map``. If this
        value is <= 0, all decoded candidates are kept.
    selection_mode:
        Unsupervised candidate-best selection rule. ``energy_lwe_prior`` is the
        recommended default for the LWE/BDD experiments.
    expected_secret_weight_ratio:
        Expected nonzero ratio of the ternary secret. Set to a negative value to
        disable the secret-weight-gap term.
    expected_noise_rate:
        Expected nonzero ratio of the sparse ternary noise. Set to a negative
        value to disable the noise-weight-gap term.
    prior_weight_*:
        Weights in the LWE-prior score. Large violation weights enforce ternary
        secret/noise structure; smaller terms tune weight matching, noise norm,
        local energy, and measurement support.
    energy_match_tol:
        Tolerance for matching candidate local energy to an exact local oracle.
    decoded_top_states_to_store:
        Number of decoded states to keep in lightweight CSV/debug outputs.
    """

    candidate_topk: int = 20
    selection_mode: CandidateSelectionMode = "energy_lwe_prior"

    expected_secret_weight_ratio: float = 0.5
    expected_noise_rate: float = 0.25

    prior_weight_secret_violation: float = 1000.0
    prior_weight_noise_violation: float = 1000.0
    prior_weight_secret_weight_gap: float = 20.0
    prior_weight_noise_weight_gap: float = 10.0
    prior_weight_noise_l2: float = 1.0
    prior_weight_local_energy: float = 0.05
    prior_weight_count_log: float = 0.1

    energy_match_tol: float = 1e-9
    decoded_top_states_to_store: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRerankResult:
    """Output of VarQITE candidate generation and reranking."""

    candidates: list[dict[str, Any]]
    best_candidate: dict[str, Any]
    primary_candidate: dict[str, Any]
    decoded_top_states: list[tuple[str, int]]

    candidate_num_unique: int
    candidate_topk_used: int

    # Diagnostic upper bounds; these use true-instance information when present.
    any_secret_success: bool
    any_geometry_improved: bool
    any_matches_exact_state: bool
    any_matches_exact_energy: bool
    min_dist2_true_among_candidates: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def vec_to_list(x: np.ndarray) -> list[int]:
    return [int(v) for v in np.asarray(x).reshape(-1)]


def normalize_lattice_vector(x: np.ndarray) -> np.ndarray:
    """Round a theoretically integral lattice vector to integer dtype."""
    return np.rint(np.asarray(x)).astype(int)


def centered_mod_q(x: np.ndarray, q: int) -> np.ndarray:
    """Convert a mod-q vector to centered representatives."""
    arr = np.asarray(x, dtype=int)
    return ((arr + q // 2) % q) - q // 2


def ternary_violation_count(x_centered: np.ndarray) -> int:
    """Count entries outside {-1, 0, 1}."""
    x_centered = np.asarray(x_centered, dtype=int).reshape(-1)
    return int(np.sum(~np.isin(x_centered, [-1, 0, 1])))


def expected_count(total_dim: int, ratio: float | None) -> int:
    """Return expected nonzero count. Negative/None ratio disables the term."""
    if ratio is None:
        return -1
    try:
        r = float(ratio)
    except Exception:
        return -1
    if r < 0:
        return -1
    return int(round(total_dim * r))


def parse_delta_key(key: Any, expected_m: int) -> tuple[int, ...] | None:
    """Parse a decoded-count-map key into a tuple[int, ...]."""
    if isinstance(key, np.ndarray):
        arr = key.reshape(-1).tolist()
    elif isinstance(key, (tuple, list)):
        arr = list(key)
    elif isinstance(key, str):
        try:
            parsed = ast.literal_eval(key)
        except Exception:
            return None
        if isinstance(parsed, (tuple, list)):
            arr = list(parsed)
        else:
            return None
    else:
        return None

    if len(arr) != expected_m:
        return None

    try:
        return tuple(int(v) for v in arr)
    except Exception:
        return None


def local_energy(D: np.ndarray, residual: np.ndarray, delta: np.ndarray) -> float:
    """Local BDD/CVP correction energy ||residual - D delta||^2."""
    diff = np.asarray(residual) - np.asarray(D) @ np.asarray(delta)
    return float(np.dot(diff, diff))


def _has_true_secret(inst: Any) -> bool:
    return hasattr(inst, "s_mod_q") and getattr(inst, "s_mod_q") is not None


def _has_true_lattice_point(inst: Any) -> bool:
    return hasattr(inst, "w_true") and getattr(inst, "w_true") is not None


def compute_lwe_prior_features(
    *,
    A: np.ndarray,
    c_centered: np.ndarray,
    s_hat_mod_q: np.ndarray,
    q: int,
    local_energy_value: float,
    count: int,
    config: CandidateRerankConfig,
) -> dict[str, Any]:
    """Compute LWE-prior features and score for a candidate secret.

    This function does not use the true secret. It only uses public ``A``, ``c``,
    ``q`` and distributional assumptions on the secret/noise.
    """
    A = np.asarray(A, dtype=int)
    c_centered = np.asarray(c_centered, dtype=int).reshape(-1)
    s_hat_mod_q = np.asarray(s_hat_mod_q, dtype=int).reshape(-1) % int(q)

    s_centered = centered_mod_q(s_hat_mod_q, int(q))

    pred = (A @ s_hat_mod_q) % int(q)
    c_mod_q = c_centered % int(q)
    e_hat_centered = centered_mod_q(c_mod_q - pred, int(q))

    secret_violation = ternary_violation_count(s_centered)
    noise_violation = ternary_violation_count(e_hat_centered)

    secret_weight = int(np.sum(s_centered != 0))
    noise_weight = int(np.sum(e_hat_centered != 0))

    target_secret_weight = expected_count(
        len(s_centered), config.expected_secret_weight_ratio
    )
    target_noise_weight = expected_count(
        len(e_hat_centered), config.expected_noise_rate
    )

    secret_weight_gap = 0 if target_secret_weight < 0 else abs(secret_weight - target_secret_weight)
    noise_weight_gap = 0 if target_noise_weight < 0 else abs(noise_weight - target_noise_weight)
    noise_l2 = float(np.dot(e_hat_centered, e_hat_centered))

    score = (
        config.prior_weight_secret_violation * secret_violation
        + config.prior_weight_noise_violation * noise_violation
        + config.prior_weight_secret_weight_gap * secret_weight_gap
        + config.prior_weight_noise_weight_gap * noise_weight_gap
        + config.prior_weight_noise_l2 * noise_l2
        + config.prior_weight_local_energy * float(local_energy_value)
        - config.prior_weight_count_log * np.log1p(max(0, int(count)))
    )

    return {
        "s_centered": vec_to_list(s_centered),
        "e_hat_centered": vec_to_list(e_hat_centered),
        "secret_ternary_violation": int(secret_violation),
        "noise_ternary_violation": int(noise_violation),
        "secret_weight": int(secret_weight),
        "target_secret_weight": int(target_secret_weight),
        "secret_weight_gap": int(secret_weight_gap),
        "noise_weight": int(noise_weight),
        "target_noise_weight": int(target_noise_weight),
        "noise_weight_gap": int(noise_weight_gap),
        "noise_l2": float(noise_l2),
        "lwe_prior_score": float(score),
    }


def evaluate_delta_candidate(
    *,
    inst: Any,
    D: np.ndarray,
    w_babai: np.ndarray,
    residual: np.ndarray,
    delta: np.ndarray,
    count: int,
    total_count: int,
    q: int,
    config: CandidateRerankConfig,
    delta_exact: np.ndarray | None = None,
    exact_best_energy: float | None = None,
) -> dict[str, Any]:
    """Evaluate one local correction candidate.

    The fields used by the selection rules do not use the true secret. If the
    supplied ``inst`` contains ``s_mod_q`` or ``w_true``, diagnostic fields are
    also filled for experiments.
    """
    delta = np.asarray(delta, dtype=int).reshape(-1)
    D = np.asarray(D)
    w_babai = np.asarray(w_babai)
    residual = np.asarray(residual)

    w_candidate = normalize_lattice_vector(w_babai + D @ delta)
    s_hat = recover_secret_from_lattice_point(inst.A, w_candidate, int(q))
    energy = local_energy(D, residual, delta)
    prob = float(count / total_count) if int(total_count) > 0 else 0.0

    secret_success = None
    if _has_true_secret(inst):
        secret_success = bool(np.array_equal(s_hat % int(q), inst.s_mod_q % int(q)))

    dist2_true = None
    geometry_improved = None
    if _has_true_lattice_point(inst):
        w_true = np.asarray(inst.w_true, dtype=int)
        dist2_true = int(np.sum((w_true - w_candidate) ** 2))
        dist2_babai = int(np.sum((w_true - normalize_lattice_vector(w_babai)) ** 2))
        geometry_improved = bool(dist2_true < dist2_babai)

    matches_exact_state = False
    if delta_exact is not None:
        matches_exact_state = bool(np.array_equal(delta, np.asarray(delta_exact, dtype=int)))

    matches_exact_energy = False
    if exact_best_energy is not None:
        matches_exact_energy = bool(abs(float(energy) - float(exact_best_energy)) <= config.energy_match_tol)

    prior_features = compute_lwe_prior_features(
        A=np.asarray(inst.A, dtype=int),
        c_centered=np.asarray(inst.c_centered, dtype=int),
        s_hat_mod_q=s_hat,
        q=int(q),
        local_energy_value=float(energy),
        count=int(count),
        config=config,
    )

    out = {
        "delta": vec_to_list(delta),
        "count": int(count),
        "prob": float(prob),
        "energy": float(energy),
        "s_hat": vec_to_list(s_hat),
        "secret_success": secret_success,
        "w_candidate": vec_to_list(w_candidate),
        "dist2_true": dist2_true,
        "geometry_improved": geometry_improved,
        "matches_exact_state": bool(matches_exact_state),
        "matches_exact_energy": bool(matches_exact_energy),
        "l1_norm": int(np.sum(np.abs(delta))),
        "l0_norm": int(np.sum(delta != 0)),
    }
    out.update(prior_features)
    return out


def select_candidate_best(
    candidates: list[dict[str, Any]],
    mode: CandidateSelectionMode,
) -> dict[str, Any]:
    """Select candidate-best using only unsupervised candidate features."""
    if not candidates:
        raise ValueError("candidate list is empty")

    mode = str(mode)
    if mode == "count":
        key_fn = lambda c: (-int(c["count"]), float(c["energy"]), int(c["l1_norm"]))
    elif mode == "count_then_energy":
        key_fn = lambda c: (-int(c["count"]), float(c["energy"]), int(c["l1_norm"]))
    elif mode == "energy":
        key_fn = lambda c: (float(c["energy"]), int(c["l1_norm"]), -int(c["count"]))
    elif mode == "sparse_energy_then_count":
        key_fn = lambda c: (float(c["energy"]), int(c["l1_norm"]), -int(c["count"]))
    elif mode == "energy_then_count":
        key_fn = lambda c: (float(c["energy"]), -int(c["count"]), int(c["l1_norm"]))
    elif mode == "lwe_prior":
        key_fn = lambda c: (
            float(c.get("lwe_prior_score", 1e18)),
            int(c.get("secret_ternary_violation", 999)),
            int(c.get("noise_ternary_violation", 999)),
            int(c.get("noise_weight_gap", 999)),
            -int(c["count"]),
            float(c["energy"]),
        )
    elif mode == "energy_lwe_prior":
        key_fn = lambda c: (
            float(c.get("lwe_prior_score", 1e18)),
            float(c["energy"]),
            -int(c["count"]),
            int(c["l1_norm"]),
        )
    else:
        raise ValueError(f"Unknown candidate selection mode: {mode}")

    return min(candidates, key=key_fn)


def build_qite_candidates(
    *,
    qite_result: Any,
    inst: Any,
    D: np.ndarray,
    w_babai: np.ndarray,
    residual: np.ndarray,
    q: int,
    config: CandidateRerankConfig | None = None,
    delta_exact: np.ndarray | None = None,
    exact_best_energy: float | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    """Build evaluated correction candidates from VarQITE decoded samples."""
    config = config or CandidateRerankConfig()
    expected_m = int(np.asarray(D).shape[1])
    raw_items: list[tuple[tuple[int, ...], int]] = []

    decoded_map = getattr(qite_result, "decoded_count_map", None)
    if decoded_map:
        for k, v in decoded_map.items():
            delta_tuple = parse_delta_key(k, expected_m=expected_m)
            if delta_tuple is not None:
                raw_items.append((delta_tuple, int(v)))

    primary_tuple = tuple(int(v) for v in np.asarray(qite_result.delta_qite, dtype=int).reshape(-1))
    if not raw_items:
        raw_items.append((primary_tuple, 0))
    else:
        existing = {x[0] for x in raw_items}
        if primary_tuple not in existing:
            raw_items.append((primary_tuple, 0))

    count_by_delta: dict[tuple[int, ...], int] = {}
    for delta_tuple, count in raw_items:
        count_by_delta[delta_tuple] = count_by_delta.get(delta_tuple, 0) + int(count)

    sorted_items = sorted(count_by_delta.items(), key=lambda kv: -kv[1])
    total_count = int(sum(count_by_delta.values()))

    if int(config.candidate_topk) > 0:
        selected_items = sorted_items[: int(config.candidate_topk)]
    else:
        selected_items = sorted_items

    candidates: list[dict[str, Any]] = []
    for delta_tuple, count in selected_items:
        candidates.append(
            evaluate_delta_candidate(
                inst=inst,
                D=D,
                w_babai=w_babai,
                residual=residual,
                delta=np.asarray(delta_tuple, dtype=int),
                count=int(count),
                total_count=total_count,
                q=int(q),
                config=config,
                delta_exact=delta_exact,
                exact_best_energy=exact_best_energy,
            )
        )

    n_store = max(0, int(config.decoded_top_states_to_store))
    decoded_top_states = [(str(tuple(delta)), int(count)) for delta, count in selected_items[:n_store]]
    return candidates, decoded_top_states


def rerank_varqite_candidates(
    *,
    qite_result: Any,
    inst: Any,
    D: np.ndarray,
    w_babai: np.ndarray,
    residual: np.ndarray,
    q: int,
    config: CandidateRerankConfig | None = None,
    delta_exact: np.ndarray | None = None,
    exact_best_energy: float | None = None,
) -> CandidateRerankResult:
    """Generate VarQITE candidates and select the LWE-prior candidate-best.

    This is the main interface intended for experiment scripts.
    """
    config = config or CandidateRerankConfig()
    candidates, decoded_top_states = build_qite_candidates(
        qite_result=qite_result,
        inst=inst,
        D=D,
        w_babai=w_babai,
        residual=residual,
        q=int(q),
        config=config,
        delta_exact=delta_exact,
        exact_best_energy=exact_best_energy,
    )

    if not candidates:
        raise ValueError("No candidate could be built from VarQITE result.")

    best = select_candidate_best(candidates, config.selection_mode)

    primary_delta = vec_to_list(np.asarray(qite_result.delta_qite, dtype=int))
    primary = next((c for c in candidates if c["delta"] == primary_delta), None)
    if primary is None:
        primary = evaluate_delta_candidate(
            inst=inst,
            D=D,
            w_babai=w_babai,
            residual=residual,
            delta=np.asarray(qite_result.delta_qite, dtype=int),
            count=0,
            total_count=0,
            q=int(q),
            config=config,
            delta_exact=delta_exact,
            exact_best_energy=exact_best_energy,
        )

    secret_flags = [c.get("secret_success") for c in candidates]
    geom_flags = [c.get("geometry_improved") for c in candidates]
    dist_values = [c.get("dist2_true") for c in candidates if c.get("dist2_true") is not None]

    any_secret_success = bool(any(v is True for v in secret_flags))
    any_geometry_improved = bool(any(v is True for v in geom_flags))
    any_matches_exact_state = bool(any(c.get("matches_exact_state") is True for c in candidates))
    any_matches_exact_energy = bool(any(c.get("matches_exact_energy") is True for c in candidates))
    min_dist2 = int(min(dist_values)) if dist_values else None

    return CandidateRerankResult(
        candidates=candidates,
        best_candidate=best,
        primary_candidate=primary,
        decoded_top_states=decoded_top_states,
        candidate_num_unique=int(len(candidates)),
        candidate_topk_used=int(len(candidates)),
        any_secret_success=any_secret_success,
        any_geometry_improved=any_geometry_improved,
        any_matches_exact_state=any_matches_exact_state,
        any_matches_exact_energy=any_matches_exact_energy,
        min_dist2_true_among_candidates=min_dist2,
    )
