# -*- coding: utf-8 -*-
"""
Classical local candidate generation + LWE-prior reranking.

This module is a fair classical counterpart to:
    VarQITE candidate generation + LWE-prior reranking.

The shared reranking score is imported from:
    algorithms.solvers.quantum.qite.candidates

Default classical candidate pool
--------------------------------
The default pool is deliberately finite-budget and local:
    1. zero correction delta = 0;
    2. every accepted state along the greedy local-search trajectory;
    3. all single-coordinate one-step neighbors of zero, e.g. +/-1 in one
       local coordinate for the default {-1, 0, 1} correction space.

The raw pool is deduplicated, ranked by local energy, and truncated to the
same ``candidate_topk`` budget used by the VarQITE candidate method. Zero
correction and the greedy final state are force-retained whenever possible.

The final candidate-best selection uses exactly the same LWE-prior score and
selection rule as the VarQITE candidate method.

Important fairness note
-----------------------
Classical candidates do not carry a measurement-count signal. Therefore the
candidate ``count`` passed to the shared score is zero. To compare both methods
without any count-based advantage, set ``prior_weight_count_log = 0.0`` for both
classical and VarQITE reranking in the experiment configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from algorithms.solvers.quantum.qite.candidates import (
    CandidateRerankConfig,
    evaluate_delta_candidate,
    select_candidate_best,
    vec_to_list,
)

ClassicalCandidatePreselectionMode = Literal[
    "energy_then_l1",
    "energy_then_l0",
    "l1_then_energy",
]


@dataclass
class ClassicalCandidateGenerationConfig:
    """Configuration for the classical local candidate pool."""

    include_zero: bool = True
    include_greedy_trajectory: bool = True
    include_single_coordinate_neighbors: bool = True

    # Force-retain these anchors whenever candidate_topk permits.
    force_include_zero: bool = True
    force_include_greedy_final: bool = True

    # Default local correction space.
    local_values: tuple[int, ...] = (-1, 0, 1)

    # Preselection before LWE-prior reranking.
    preselection_mode: ClassicalCandidatePreselectionMode = "energy_then_l1"

    # Debug / CSV preview.
    preview_states_to_store: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassicalCandidateRerankResult:
    """Output of classical candidate generation + LWE-prior reranking."""

    candidates: list[dict[str, Any]]
    best_candidate: dict[str, Any]
    greedy_final_candidate: dict[str, Any]

    candidate_preview_states: list[tuple[str, float]]
    candidate_num_generated_raw: int
    candidate_topk_used: int

    # Diagnostic upper bounds; these use true-instance information when present.
    any_secret_success: bool
    any_geometry_improved: bool
    any_matches_exact_state: bool
    any_matches_exact_energy: bool
    min_dist2_true_among_candidates: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_local_values(values: tuple[int, ...]) -> tuple[int, ...]:
    vals = tuple(int(v) for v in values)
    if 0 not in vals:
        raise ValueError("local_values must contain 0.")
    if len(set(vals)) != len(vals):
        raise ValueError(f"local_values contains duplicates: {vals}")
    return vals


def _delta_tuple(delta: np.ndarray | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(delta, dtype=int).reshape(-1))


def _add_candidate(
    source_map: dict[tuple[int, ...], set[str]],
    delta: np.ndarray | list[int] | tuple[int, ...],
    source: str,
    expected_m: int,
) -> None:
    key = _delta_tuple(delta)
    if len(key) != int(expected_m):
        raise ValueError(
            f"Classical candidate dimension mismatch: expected {expected_m}, got {len(key)}."
        )
    source_map.setdefault(key, set()).add(str(source))


def _build_raw_candidate_sources(
    *,
    D: np.ndarray,
    greedy_result: Any,
    generation_config: ClassicalCandidateGenerationConfig,
) -> dict[tuple[int, ...], set[str]]:
    D = np.asarray(D)
    expected_m = int(D.shape[1])
    local_values = _validate_local_values(generation_config.local_values)

    source_map: dict[tuple[int, ...], set[str]] = {}
    zero = np.zeros(expected_m, dtype=int)

    if generation_config.include_zero:
        _add_candidate(source_map, zero, "zero", expected_m)

    if generation_config.include_greedy_trajectory:
        history = getattr(greedy_result, "history", []) or []
        for item in history:
            if "delta" not in item:
                continue
            _add_candidate(
                source_map,
                np.asarray(item["delta"], dtype=int),
                "greedy_trajectory",
                expected_m,
            )

    if hasattr(greedy_result, "delta_greedy"):
        _add_candidate(
            source_map,
            np.asarray(greedy_result.delta_greedy, dtype=int),
            "greedy_final",
            expected_m,
        )

    if generation_config.include_single_coordinate_neighbors:
        nonzero_values = [int(v) for v in local_values if int(v) != 0]
        for i in range(expected_m):
            for value in nonzero_values:
                delta = np.zeros(expected_m, dtype=int)
                delta[i] = int(value)
                _add_candidate(
                    source_map,
                    delta,
                    "single_coordinate_neighbor",
                    expected_m,
                )

    if not source_map:
        _add_candidate(source_map, zero, "fallback_zero", expected_m)

    return source_map


def _preselection_key(
    candidate: dict[str, Any],
    mode: ClassicalCandidatePreselectionMode,
) -> tuple[Any, ...]:
    energy = float(candidate["energy"])
    l1 = int(candidate.get("l1_norm", 10**9))
    l0 = int(candidate.get("l0_norm", 10**9))
    delta_tuple = tuple(int(v) for v in candidate["delta"])

    if mode == "energy_then_l1":
        return (energy, l1, l0, delta_tuple)
    if mode == "energy_then_l0":
        return (energy, l0, l1, delta_tuple)
    if mode == "l1_then_energy":
        return (l1, energy, l0, delta_tuple)
    raise ValueError(f"Unknown classical candidate preselection mode: {mode}")


def rerank_classical_local_candidates(
    *,
    inst: Any,
    D: np.ndarray,
    w_babai: np.ndarray,
    residual: np.ndarray,
    q: int,
    greedy_result: Any,
    rerank_config: CandidateRerankConfig | None = None,
    generation_config: ClassicalCandidateGenerationConfig | None = None,
    delta_exact: np.ndarray | None = None,
    exact_best_energy: float | None = None,
) -> ClassicalCandidateRerankResult:
    """Generate a classical local candidate pool and apply the shared LWE prior."""
    rerank_config = rerank_config or CandidateRerankConfig()
    generation_config = generation_config or ClassicalCandidateGenerationConfig()

    D = np.asarray(D)
    expected_m = int(D.shape[1])

    source_map = _build_raw_candidate_sources(
        D=D,
        greedy_result=greedy_result,
        generation_config=generation_config,
    )

    raw_candidates: list[dict[str, Any]] = []
    for delta_key, sources in source_map.items():
        candidate = evaluate_delta_candidate(
            inst=inst,
            D=D,
            w_babai=w_babai,
            residual=residual,
            delta=np.asarray(delta_key, dtype=int),
            count=0,
            total_count=0,
            q=int(q),
            config=rerank_config,
            delta_exact=delta_exact,
            exact_best_energy=exact_best_energy,
        )
        candidate["classical_sources"] = sorted(str(s) for s in sources)
        raw_candidates.append(candidate)

    raw_candidates.sort(
        key=lambda cand: _preselection_key(cand, generation_config.preselection_mode)
    )

    candidate_topk = int(rerank_config.candidate_topk)
    if candidate_topk <= 0:
        selected = list(raw_candidates)
    else:
        forced_delta_order: list[list[int]] = []
        if generation_config.force_include_zero:
            forced_delta_order.append([0] * expected_m)
        if generation_config.force_include_greedy_final and hasattr(greedy_result, "delta_greedy"):
            forced_delta_order.append(
                vec_to_list(np.asarray(greedy_result.delta_greedy, dtype=int))
            )

        raw_by_delta = {
            tuple(int(v) for v in cand["delta"]): cand for cand in raw_candidates
        }
        selected: list[dict[str, Any]] = []
        selected_keys: set[tuple[int, ...]] = set()

        for delta_list in forced_delta_order:
            key = tuple(int(v) for v in delta_list)
            if key in raw_by_delta and key not in selected_keys and len(selected) < candidate_topk:
                selected.append(raw_by_delta[key])
                selected_keys.add(key)

        for cand in raw_candidates:
            if len(selected) >= candidate_topk:
                break
            key = tuple(int(v) for v in cand["delta"])
            if key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(key)

    if not selected:
        raise ValueError("No classical candidate retained after preselection.")

    best = select_candidate_best(selected, rerank_config.selection_mode)

    greedy_delta = vec_to_list(np.asarray(greedy_result.delta_greedy, dtype=int))
    greedy_final = next((c for c in selected if c["delta"] == greedy_delta), None)
    if greedy_final is None:
        greedy_final = evaluate_delta_candidate(
            inst=inst,
            D=D,
            w_babai=w_babai,
            residual=residual,
            delta=np.asarray(greedy_result.delta_greedy, dtype=int),
            count=0,
            total_count=0,
            q=int(q),
            config=rerank_config,
            delta_exact=delta_exact,
            exact_best_energy=exact_best_energy,
        )
        greedy_final["classical_sources"] = ["greedy_final_outside_selected_pool"]

    secret_flags = [c.get("secret_success") for c in selected]
    geom_flags = [c.get("geometry_improved") for c in selected]
    dist_values = [c.get("dist2_true") for c in selected if c.get("dist2_true") is not None]

    any_secret_success = bool(any(v is True for v in secret_flags))
    any_geometry_improved = bool(any(v is True for v in geom_flags))
    any_matches_exact_state = bool(any(c.get("matches_exact_state") is True for c in selected))
    any_matches_exact_energy = bool(any(c.get("matches_exact_energy") is True for c in selected))
    min_dist2 = int(min(dist_values)) if dist_values else None

    preview_n = max(0, int(generation_config.preview_states_to_store))
    preview = [
        (str(tuple(int(v) for v in c["delta"])), float(c["energy"]))
        for c in selected[:preview_n]
    ]

    return ClassicalCandidateRerankResult(
        candidates=selected,
        best_candidate=best,
        greedy_final_candidate=greedy_final,
        candidate_preview_states=preview,
        candidate_num_generated_raw=int(len(raw_candidates)),
        candidate_topk_used=int(len(selected)),
        any_secret_success=any_secret_success,
        any_geometry_improved=any_geometry_improved,
        any_matches_exact_state=any_matches_exact_state,
        any_matches_exact_energy=any_matches_exact_energy,
        min_dist2_true_among_candidates=min_dist2,
    )
