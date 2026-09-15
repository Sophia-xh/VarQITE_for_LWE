# -*- coding: utf-8 -*-
"""
run_figure4_case_study.py
=========================

End-to-end single-instance mechanism analysis for:
    VarQITE candidates + LWE-prior reranking

This is the script that produced the archived Figure 4 data
(data/figure4_mechanism_analysis/) and the paper's Figure 4.

What this script does for one LWE instance:
  1. Regenerates the LWE instance from (n, m, q, seed).
  2. Runs Babai, exact local oracle (optional), greedy local descent,
     VarQITE, and VarQITE candidate reranking.
  3. Builds a greedy-trajectory candidate pool using the same LWE-prior score.
  4. Measures the two-parameter energy surface shown in panel (b).
  5. Saves the CSV/JSON logs and produces the four-panel figure
     figure4_mechanism_analysis.{eps,pdf,png} plus two byproduct figures.

Reproducing Figure 4 does NOT require this script: it repeats the shot-based
VarQITE simulation, so its energy trajectory and surface differ from the
archived run within shot noise. Use
    scripts/case_study/plot_figure4_mechanism_analysis.py
to redraw the archived figure. Running this script is the only way to obtain a
panel (b) energy surface, which the archive does not contain.

Example, reproducing the archived instance:
    python scripts/case_study/run_figure4_case_study.py \
        --n 6 --m 8 --q 17 --seed 1 --steps 30

To pick an instance automatically from an existing main-experiment run, where
Babai fails but VarQITE candidates + LWE-prior succeeds:
    python scripts/case_study/run_figure4_case_study.py \
        --source-run-dir data/figure3_performance_comparison/main_experiment \
        --auto-pick --n 8 --steps 20 --seed 1
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Avoid BLAS oversubscription when pyqpanda / numpy internally uses threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator

SCIS_FONT_SIZE = 8
SCIS_PNG_DPI = 600
SCIS_LINEWIDTH_PT = 0.6
SCIS_MARKER_EDGEWIDTH_PT = 0.6

# SCIS2026.cls sets the full text width to 176 mm.
# Create the two full-width composite figures at their final paper width so
# LaTeX does not need to shrink them and the plotted text remains 8 pt.
SCIS_TEXT_WIDTH_MM = 176.0
SCIS_TEXT_WIDTH_IN = SCIS_TEXT_WIDTH_MM / 25.4
FIG_1_SIZE_IN = (
    SCIS_TEXT_WIDTH_IN,
    SCIS_TEXT_WIDTH_IN * 4.15 / 10.6,
)
FIG_2_4_SIZE_IN = (
    SCIS_TEXT_WIDTH_IN,
    5.45,
)

# Final size of the merged 2 x 2 mechanism-analysis figure.
# Width remains exactly the SCIS full-text width (176 mm), while the
# 5.70-inch height keeps the four panels compact without shrinking the
# 8-pt text used throughout the manuscript.
FIG_MECHANISM_4PANEL_SIZE_IN = (
    SCIS_TEXT_WIDTH_IN,
    5.70,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": SCIS_FONT_SIZE,
        "axes.titlesize": SCIS_FONT_SIZE,
        "axes.labelsize": SCIS_FONT_SIZE,
        "xtick.labelsize": SCIS_FONT_SIZE,
        "ytick.labelsize": SCIS_FONT_SIZE,
        "legend.fontsize": SCIS_FONT_SIZE,
        "lines.linewidth": SCIS_LINEWIDTH_PT,
        "axes.linewidth": SCIS_LINEWIDTH_PT,
        "grid.linewidth": SCIS_LINEWIDTH_PT,
        "xtick.major.width": SCIS_LINEWIDTH_PT,
        "ytick.major.width": SCIS_LINEWIDTH_PT,
        "xtick.minor.width": SCIS_LINEWIDTH_PT,
        "ytick.minor.width": SCIS_LINEWIDTH_PT,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "savefig.dpi": SCIS_PNG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.interpolate import splprep, splev
except Exception:  # pragma: no cover
    splprep = None
    splev = None


# ----------------------------------------------------------------------
# Project-root detection.
# This script is designed to be placed in scripts/case_study/.
# It also works if executed from another folder, as long as an ancestor
# contains the algorithms/ package.
# ----------------------------------------------------------------------
def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "algorithms").exists():
            return p
    # fallback for the intended location: <root>/scripts/case_study/script.py
    return here.parent.parent.parent


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.instance.lwe_instance import generate_small_lwe_instance
from algorithms.lattice.hnf_basis import full_rank_basis_from_hnf
from algorithms.lattice.reduction import reduce_basis
from algorithms.lattice.babai import babai_nearest_plane
from algorithms.local_model.local_hamiltonian import build_diagonal_hamiltonian
from algorithms.recovery.recover_secret import recover_secret_from_lattice_point
from algorithms.solvers.classical.greedy_local_solver import greedy_local_from_babai
from algorithms.solvers.classical.candidates import (
    ClassicalCandidateGenerationConfig,
    rerank_classical_local_candidates,
)
from algorithms.solvers.quantum.qite import (
    qite_optimize_local_pyqpanda3,
    CandidateRerankConfig,
    rerank_varqite_candidates,
)
from algorithms.solvers.quantum.qite.ansatz import build_ry_chain_ansatz_with_biased_init
from algorithms.solvers.quantum.common.measurement import measured_energy
from algorithms.solvers.quantum.common.backend import _fresh_qvm, _finalize_qvm


# =========================================================
# Generic helpers
# =========================================================
def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def vec_to_list(x: Any) -> list[int]:
    return [int(v) for v in np.asarray(x).reshape(-1)]


def normalize_lattice_vector(x: Any) -> np.ndarray:
    return np.rint(np.asarray(x)).astype(int)


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", ""}:
        return False
    return bool(x)


def round_m_from_ratio(n: int, c_star: float, mode: str) -> int:
    x = float(c_star) * int(n)
    if mode == "ceil":
        return int(math.ceil(x))
    if mode == "floor":
        return int(math.floor(x))
    if mode == "round":
        return int(round(x))
    raise ValueError(f"Unsupported m_rounding: {mode}")


def make_secret_probs(weight_total: float) -> tuple[float, float, float]:
    w = float(weight_total)
    if not (0.0 <= w <= 1.0):
        raise ValueError("secret_weight_total must be in [0, 1].")
    return (w / 2.0, 1.0 - w, w / 2.0)


def make_noise_probs(error_rate: float) -> tuple[float, float, float]:
    tau = float(error_rate)
    if not (0.0 <= tau <= 1.0):
        raise ValueError("noise_error_rate must be in [0, 1].")
    return (tau / 2.0, 1.0 - tau, tau / 2.0)


def safe_generate_instance(n: int, m: int, q: int, seed: int, args: argparse.Namespace):
    """Generate LWE instance while staying compatible with older signatures."""
    sig = inspect.signature(generate_small_lwe_instance)
    kwargs: dict[str, Any] = {"n": n, "m": m, "q": q, "seed": seed}

    if "secret_probs" in sig.parameters:
        kwargs["secret_probs"] = make_secret_probs(float(args.secret_weight_total))
    if "noise_probs" in sig.parameters:
        kwargs["noise_probs"] = make_noise_probs(float(args.noise_error_rate))
    if "secret_mode" in sig.parameters and args.secret_mode is not None:
        kwargs["secret_mode"] = args.secret_mode
    if "noise_mode" in sig.parameters and args.noise_mode is not None:
        kwargs["noise_mode"] = args.noise_mode
    if "require_nonzero_secret" in sig.parameters:
        kwargs["require_nonzero_secret"] = bool(args.require_nonzero_secret)

    inst = generate_small_lwe_instance(**kwargs)
    if bool(args.require_nonzero_secret) and np.all(np.asarray(inst.s_centered) == 0):
        raise RuntimeError("Generated an all-zero secret although require_nonzero_secret=True.")
    return inst


def make_candidate_config(args: argparse.Namespace) -> CandidateRerankConfig:
    return CandidateRerankConfig(
        candidate_topk=int(args.candidate_topk),
        selection_mode=str(args.candidate_selection),
        expected_secret_weight_ratio=float(args.expected_secret_weight_ratio),
        expected_noise_rate=float(args.expected_noise_rate),
        prior_weight_secret_violation=float(args.prior_weight_secret_violation),
        prior_weight_noise_violation=float(args.prior_weight_noise_violation),
        prior_weight_secret_weight_gap=float(args.prior_weight_secret_weight_gap),
        prior_weight_noise_weight_gap=float(args.prior_weight_noise_weight_gap),
        prior_weight_noise_l2=float(args.prior_weight_noise_l2),
        prior_weight_local_energy=float(args.prior_weight_local_energy),
        prior_weight_count_log=float(args.prior_weight_count_log),
        energy_match_tol=float(args.candidate_energy_match_tol),
        decoded_top_states_to_store=int(args.decoded_top_states_to_store),
    )


def make_classical_generation_config(args: argparse.Namespace) -> ClassicalCandidateGenerationConfig:
    return ClassicalCandidateGenerationConfig(
        include_zero=True,
        include_greedy_trajectory=True,
        include_single_coordinate_neighbors=bool(args.include_single_coordinate_neighbors),
        force_include_zero=True,
        force_include_greedy_final=True,
        local_values=tuple(int(v) for v in args.local_values),
        preselection_mode=str(args.classical_preselection_mode),
        preview_states_to_store=int(args.classical_preview_states_to_store),
    )


def local_energy(D: np.ndarray, residual: np.ndarray, delta: np.ndarray) -> float:
    diff = np.asarray(residual, dtype=float) - np.asarray(D, dtype=float) @ np.asarray(delta, dtype=float)
    return float(np.dot(diff, diff))


def json_dump(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def dataframe_to_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def serialize_for_csv(x: Any) -> Any:
    if isinstance(x, (list, tuple, dict, np.ndarray)):
        return json.dumps(to_jsonable(x), ensure_ascii=False)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def candidates_to_df(candidates: list[dict[str, Any]], source: str, selected_delta: list[int] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_tuple = tuple(selected_delta) if selected_delta is not None else None

    for idx, cand in enumerate(candidates):
        row = dict(cand)
        row["candidate_id"] = idx
        row["source"] = source
        delta_tuple = tuple(int(v) for v in cand.get("delta", []))
        row["delta_key"] = str(delta_tuple)
        row["is_selected"] = bool(selected_tuple is not None and delta_tuple == selected_tuple)
        row["is_energy_min"] = False
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0 and "energy" in df.columns:
        min_idx = pd.to_numeric(df["energy"], errors="coerce").idxmin()
        if not pd.isna(min_idx):
            df.loc[min_idx, "is_energy_min"] = True
        df["rank_by_energy"] = pd.to_numeric(df["energy"], errors="coerce").rank(method="first", ascending=True).astype(int)
    if len(df) > 0 and "lwe_prior_score" in df.columns:
        df["rank_by_lwe_prior"] = pd.to_numeric(df["lwe_prior_score"], errors="coerce").rank(method="first", ascending=True).astype(int)
    if len(df) > 0 and "count" in df.columns:
        df["rank_by_count"] = pd.to_numeric(df["count"], errors="coerce").rank(method="first", ascending=False).astype(int)

    for col in df.columns:
        df[col] = df[col].map(serialize_for_csv)
    return df


def save_figure(fig: plt.Figure, out_base: Path, use_tight_layout: bool = True) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        fig.tight_layout()

    # Do not use bbox_inches="tight" here. Keeping the declared figsize makes
    # the PDF's physical width predictable, which is necessary for preserving
    # the intended 8 pt text size after insertion into the SCIS manuscript.
    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=SCIS_PNG_DPI,
    )
    fig.savefig(
        out_base.with_suffix(".pdf"),
    )
    fig.savefig(
        out_base.with_suffix(".eps"),
        format="eps",
    )
    plt.close(fig)


def apply_scis_axis_style(ax: plt.Axes, *, grid: bool = True, grid_axis: str = "both") -> None:
    """Apply SCIS-friendly axis styling."""
    ax.tick_params(axis="both", labelsize=SCIS_FONT_SIZE, width=SCIS_LINEWIDTH_PT, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(SCIS_LINEWIDTH_PT)
    if grid:
        ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=SCIS_LINEWIDTH_PT)


def _smooth_grid_2d_numpy(Z: np.ndarray, passes: int = 2) -> np.ndarray:
    """Small dependency-free smoothing for a measured energy grid."""
    Z = np.asarray(Z, dtype=float).copy()
    passes = max(0, int(passes))
    for _ in range(passes):
        P = np.pad(Z, pad_width=1, mode="edge")
        Z = (
            4.0 * P[1:-1, 1:-1]
            + 2.0 * (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:])
            + (P[:-2, :-2] + P[:-2, 2:] + P[2:, :-2] + P[2:, 2:])
        ) / 16.0
    return Z


def _bilinear_interpolate_regular_grid(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, xq, yq) -> np.ndarray:
    """Bilinear interpolation on a regular grid. Query points are clipped to the grid box."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    Z = np.asarray(Z, dtype=float)
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)

    xq = np.clip(xq, xs[0], xs[-1])
    yq = np.clip(yq, ys[0], ys[-1])

    ix = np.searchsorted(xs, xq, side="right") - 1
    iy = np.searchsorted(ys, yq, side="right") - 1
    ix = np.clip(ix, 0, len(xs) - 2)
    iy = np.clip(iy, 0, len(ys) - 2)

    x0 = xs[ix]
    x1 = xs[ix + 1]
    y0 = ys[iy]
    y1 = ys[iy + 1]

    tx = np.divide(xq - x0, x1 - x0, out=np.zeros_like(xq, dtype=float), where=np.abs(x1 - x0) > 1e-12)
    ty = np.divide(yq - y0, y1 - y0, out=np.zeros_like(yq, dtype=float), where=np.abs(y1 - y0) > 1e-12)

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    return (
        (1 - tx) * (1 - ty) * z00
        + tx * (1 - ty) * z10
        + (1 - tx) * ty * z01
        + tx * ty * z11
    )


def _smooth_path_for_plot(x: np.ndarray, y: np.ndarray, z: np.ndarray, n_points: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a smoother 3D plotting path. Falls back to linear interpolation if scipy is unavailable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(x) < 3:
        return x, y, z

    if splprep is not None and len(x) >= 4:
        try:
            pts = np.vstack([x, y, z])
            tck, u = splprep(pts, s=0.08)
            u_f = np.linspace(0.0, 1.0, n_points)
            xs_s, ys_s, zs_s = splev(u_f, tck)
            return np.asarray(xs_s), np.asarray(ys_s), np.asarray(zs_s)
        except Exception:
            pass

    seg = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 1e-12:
        return x, y, z
    u_f = np.linspace(0.0, t[-1], n_points)
    return np.interp(u_f, t, x), np.interp(u_f, t, y), np.interp(u_f, t, z)


def _extract_theta_trajectory_from_qite(qite_res, theta_init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the actual VarQITE parameter trajectory and its measured energies.

    The qite history stores the parameter vector after each update in `theta`.
    We prepend the initial parameter vector and use E_before of step 0 as its
    energy. The following points use E_after for each update step.
    """
    theta_points: list[np.ndarray] = []
    energy_points: list[float] = []

    history = list(getattr(qite_res, "history", []) or [])
    theta_init = np.asarray(theta_init, dtype=float).reshape(-1)

    if history and "E_before" in history[0]:
        theta_points.append(theta_init.copy())
        energy_points.append(float(history[0]["E_before"]))

    for h in history:
        if "theta" not in h:
            continue
        theta = np.asarray(h["theta"], dtype=float).reshape(-1)
        theta_points.append(theta)
        if "E_after" in h:
            energy_points.append(float(h["E_after"]))
        elif "E_before" in h:
            energy_points.append(float(h["E_before"]))
        else:
            energy_points.append(np.nan)

    theta_final = np.asarray(getattr(qite_res, "theta_final", theta_points[-1] if theta_points else theta_init), dtype=float).reshape(-1)
    if not theta_points:
        theta_points.append(theta_final)
        energy_points.append(np.nan)
    elif not np.allclose(theta_points[-1], theta_final, atol=1e-10, rtol=1e-10):
        theta_points.append(theta_final)
        energy_points.append(energy_points[-1] if energy_points else np.nan)

    return np.vstack(theta_points), np.asarray(energy_points, dtype=float)


def _pca_components_from_trajectory(X: np.ndarray, num_params: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute two PCA directions from the actual VarQITE trajectory relative to
    the final parameter vector. If the trajectory is nearly one-dimensional,
    fall back to a second coordinate direction so that a 2D plane can still be drawn.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    # Do not re-center around the trajectory mean, because we want the final
    # parameter vector to remain at the origin of the projected plane.
    try:
        _, svals, vh = np.linalg.svd(X, full_matrices=False)
    except Exception:
        svals = np.zeros(0)
        vh = np.zeros((0, num_params))

    components: list[np.ndarray] = []
    for k in range(min(2, vh.shape[0])):
        v = np.asarray(vh[k], dtype=float).reshape(-1)
        if np.linalg.norm(v) > 1e-12:
            components.append(v / np.linalg.norm(v))

    # Fallback directions if the actual trajectory does not span two stable directions.
    if len(components) < 2:
        movement = np.abs(X).max(axis=0) if X.size else np.zeros(num_params)
        order = list(np.argsort(movement)[::-1])
        for idx in order + list(range(num_params)):
            e = np.zeros(num_params, dtype=float)
            e[int(idx)] = 1.0
            if all(abs(float(np.dot(e, c))) < 0.95 for c in components):
                components.append(e)
            if len(components) >= 2:
                break

    C = np.vstack(components[:2])
    if len(svals) >= 2 and np.sum(svals ** 2) > 0:
        explained = (svals[:2] ** 2) / np.sum(svals ** 2)
    else:
        explained = np.array([np.nan, np.nan])
    return C, explained



def _select_two_real_parameter_axes_from_trajectory(theta_traj: np.ndarray) -> tuple[int, int]:
    """
    Select two REAL variational parameters for visualization.

    The selected axes are the two original theta coordinates with the largest
    movement range along the actual VarQITE trajectory. Unlike PCA axes, these
    are real circuit parameters, so the figure axes can be labeled as theta_i
    and theta_j directly.
    """
    theta_traj = np.asarray(theta_traj, dtype=float)
    if theta_traj.ndim != 2 or theta_traj.shape[1] < 2:
        raise ValueError("Need at least two real variational parameters to draw the 3D surface.")

    ranges = np.nanmax(theta_traj, axis=0) - np.nanmin(theta_traj, axis=0)
    order = np.argsort(ranges)[::-1]
    i, j = int(order[0]), int(order[1])
    return min(i, j), max(i, j)


def plot_varqite_real_parameter_energy_surface(
    ax,
    *,
    qite_res,
    D,
    residual,
    args,
):
    """
    Draw a smooth 3D energy surface over two REAL VarQITE parameters.

    This replaces the PCA-projected surface. The horizontal axes are two
    original variational parameters, theta_i and theta_j, chosen automatically
    as the two parameters with the largest movement range along the actual
    VarQITE trajectory.

    The surface is evaluated by fixing all other parameters at the final
    VarQITE parameter vector and sweeping theta_i and theta_j on a regular grid.
    The actual VarQITE parameter path is overlaid on the surface.
    """
    D_orig = np.asarray(D, dtype=float)
    r_orig = np.asarray(residual, dtype=float).reshape(-1)

    scale = max(
        1.0,
        float(np.max(np.abs(D_orig))) if D_orig.size > 0 else 0.0,
        float(np.max(np.abs(r_orig))) if r_orig.size > 0 else 0.0,
    )
    D_work = D_orig / scale
    r_work = r_orig / scale

    # Rebuild the same ansatz structure used by the VarQITE optimizer.
    vqc, _, num_params, theta_init = build_ry_chain_ansatz_with_biased_init(
        D=D_work,
        residual=r_work,
        layers=int(args.layers),
        beta=1.6,
        p_zero_floor=0.02,
        p_bias_max=0.22,
        later_layer_noise=0.02,
        seed=1234,
    )

    theta_final = np.asarray(qite_res.theta_final, dtype=float).reshape(-1)
    theta_init = np.asarray(theta_init, dtype=float).reshape(-1)

    if len(theta_final) < 2 or len(theta_final) != int(num_params):
        ax.text2D(
            0.5,
            0.5,
            "Parameter dimension is not suitable for real-parameter surface",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return None, None, None

    # Actual VarQITE path in the original parameter coordinates.
    theta_traj, energy_traj = _extract_theta_trajectory_from_qite(qite_res, theta_init)
    good = np.asarray([len(t) == len(theta_final) for t in theta_traj], dtype=bool)
    theta_traj = theta_traj[good]
    energy_traj = energy_traj[good]

    if theta_traj.shape[0] < 2:
        ax.text2D(
            0.5,
            0.5,
            "Not enough trajectory points for real-parameter surface",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return None, None, None

    # Select two original circuit parameters, not PCA directions.
    param_i, param_j = _select_two_real_parameter_axes_from_trajectory(theta_traj)

    traj_x = theta_traj[:, param_i]
    traj_y = theta_traj[:, param_j]

    # Build a grid in the real theta_i/theta_j coordinate system.
    grid_size = int(getattr(args, "real_surface_grid_size", 0) or getattr(args, "trajectory_surface_grid_size", 25) or 25)
    grid_size = max(9, grid_size)
    padding_ratio = float(getattr(args, "real_surface_padding", 0.35))

    x_min, x_max = float(np.min(traj_x)), float(np.max(traj_x))
    y_min, y_max = float(np.min(traj_y)), float(np.max(traj_y))
    x_span = max(x_max - x_min, 0.18)
    y_span = max(y_max - y_min, 0.18)
    x_pad = padding_ratio * x_span
    y_pad = padding_ratio * y_span

    xs = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    ys = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX, dtype=float)

    surface_shots = int(getattr(args, "real_surface_shots", 0) or getattr(args, "trajectory_surface_shots", 120) or 120)

    qvm = _fresh_qvm()
    try:
        # Evaluate surface. All parameters except theta_i and theta_j are fixed
        # at the final VarQITE parameters.
        for a in range(grid_size):
            for b in range(grid_size):
                theta = theta_final.copy()
                theta[param_i] = XX[a, b]
                theta[param_j] = YY[a, b]
                ZZ[a, b] = measured_energy(
                    vqc=vqc,
                    params=np.asarray(theta, dtype=float),
                    h_pauli=qite_res.h_pauli,
                    shots=surface_shots,
                    bit_reversed=True,
                    y_rot_sign=1,
                    qvm=qvm,
                )

        # If the stored trajectory energies are missing, measure them here.
        if not np.all(np.isfinite(energy_traj)):
            measured_path_energies = []
            for theta in theta_traj:
                measured_path_energies.append(
                    measured_energy(
                        vqc=vqc,
                        params=np.asarray(theta, dtype=float),
                        h_pauli=qite_res.h_pauli,
                        shots=surface_shots,
                        bit_reversed=True,
                        y_rot_sign=1,
                        qvm=qvm,
                    )
                )
            energy_traj = np.asarray(measured_path_energies, dtype=float)
    finally:
        _finalize_qvm(qvm)

    smooth_passes = int(getattr(args, "real_surface_smooth_passes", 0) or getattr(args, "trajectory_surface_smooth_passes", 2) or 2)
    ZZ_plot = _smooth_grid_2d_numpy(ZZ, passes=smooth_passes)

    # Smooth surface only: no sampled point cloud.
    surf = ax.plot_surface(
        XX,
        YY,
        ZZ_plot,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        rstride=1,
        cstride=1,
    )

    # Overlay the actual VarQITE path using the REAL trajectory energies.
    # The curve is smoothed only for visualization; it still represents the
    # true 16-dimensional trajectory points and their measured energies.
    if not np.all(np.isfinite(energy_traj)):
        measured_path_energies = []
        qvm = _fresh_qvm()
        try:
            for theta in theta_traj:
                measured_path_energies.append(
                    measured_energy(
                        vqc=vqc,
                        params=np.asarray(theta, dtype=float),
                        h_pauli=qite_res.h_pauli,
                        shots=surface_shots,
                        bit_reversed=True,
                        y_rot_sign=1,
                        qvm=qvm,
                    )
                )
        finally:
            _finalize_qvm(qvm)
        energy_traj = np.asarray(measured_path_energies, dtype=float)

    path_x_s, path_y_s, path_z_s = _smooth_path_for_plot(traj_x, traj_y, energy_traj, n_points=240)

    ax.plot(
        path_x_s,
        path_y_s,
        path_z_s,
        color="black",
        linewidth=SCIS_LINEWIDTH_PT,
        label="Actual VarQITE parameter path",
        zorder=10,
    )

    # Lift the visible markers slightly above their actual z values so they are
    # not visually hidden by the surface from this camera angle.
    marker_lift = 0.02 * max(1e-8, float(np.nanmax(ZZ_plot) - np.nanmin(ZZ_plot)))
    initial_z = float(energy_traj[0]) + marker_lift
    final_z = float(energy_traj[-1]) + marker_lift

    ax.scatter(
        [traj_x[0]],
        [traj_y[0]],
        [initial_z],
        marker="o",
        s=90,
        facecolors="green",
        # edgecolors="black",
        linewidths=SCIS_MARKER_EDGEWIDTH_PT,
        label="Initial parameters",
        depthshade=False,
        zorder=20,
    )

    ax.scatter(
        [traj_x[-1]],
        [traj_y[-1]],
        [final_z],
        marker="*",
        s=300,
        # color="#FFD84D",
        color="blue",
        edgecolors="black",
        linewidths=SCIS_MARKER_EDGEWIDTH_PT,
        label="Final parameters",
        depthshade=False,
        zorder=21,
    )

    ax.set_xlabel(
        rf"$\theta_{{{param_i}}}$",
        fontsize=SCIS_FONT_SIZE,
        labelpad=6,
    )

    ax.set_ylabel(
        rf"$\theta_{{{param_j}}}$",
        fontsize=SCIS_FONT_SIZE,
        labelpad=6,
    )

    ax.set_zlabel(
        "Energy",
        fontsize=SCIS_FONT_SIZE,
        labelpad=-2,
    )

    ax.set_title(
        rf"(b) 3D energy surface over "
        rf"$\theta_{{{param_i}}}$ and $\theta_{{{param_j}}}$",
        fontsize=SCIS_FONT_SIZE,
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.tick_params(axis="x", labelsize=SCIS_FONT_SIZE, width=SCIS_LINEWIDTH_PT, pad=1)
    ax.tick_params(axis="y", labelsize=SCIS_FONT_SIZE, width=SCIS_LINEWIDTH_PT)
    ax.tick_params(axis="z", labelsize=SCIS_FONT_SIZE, width=SCIS_LINEWIDTH_PT)

    ax.view_init(elev=25, azim=-54)

    ax.legend(
        frameon=False,
        fontsize=SCIS_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        labelspacing=0.4,
        handletextpad=0.5,
        handlelength=2.0,
        borderpad=0.2,
        markerscale=0.65,
    )

    return surf, param_i, param_j

def draw_varqite_circuit_schematic(out_dir: Path, *, m: int, layers: int) -> None:
    """
    Draw the VarQITE local circuit schematic used in this case study.
    This figure is saved directly by the case-study script.
    """
    m = int(m)
    layers = int(layers)
    num_qubits = 2 * m

    pairs_to_draw = [0, 1, m - 2, m - 1] if m >= 4 else list(range(m))
    draw_rows = []
    for i in pairs_to_draw:
        draw_rows.append((2 * i, rf"$m_{i}$"))
        draw_rows.append((2 * i + 1, rf"$p_{i}$"))

    insert_ellipsis = m >= 5

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axis("off")

    x0 = 0.8
    x_layer_start = 2.65
    x_layer_gap = 1.75
    x_measure = x_layer_start + max(layers, 1) * x_layer_gap + 0.85
    x_decode = x_measure + 1.3
    x_rerank = x_decode + 1.6

    row_gap = 0.42
    rows = []
    for idx, item in enumerate(draw_rows):
        rows.append(item)
        if insert_ellipsis and idx == 3:
            rows.append((None, r"$\vdots$"))

    y_top = 4.25
    ys = [y_top - i * row_gap for i in range(len(rows))]

    encoding_text = (
        r"Encoding: "
        r"$m_i p_i=10\rightarrow -1,\;00\rightarrow 0,\;01\rightarrow +1,\;11\rightarrow$ invalid penalty"
    )
    ax.text(0.5, 4.62, encoding_text, fontsize=10.5, ha="left", va="center")

    for y, (q_idx, label) in zip(ys, rows):
        if q_idx is None:
            ax.text(0.63, y, label, fontsize=16, ha="center", va="center")
            continue
        ax.text(0.35, y, rf"$q_{{{q_idx}}}$", fontsize=10, ha="right", va="center")
        ax.text(0.48, y, label, fontsize=10, ha="left", va="center")
        ax.plot([x0, x_rerank + 0.6], [y, y], linewidth=1.0)

    ax.text(1.55, y_top + 0.38, r"$|0\rangle^{\otimes 2m}$", fontsize=10, ha="center", va="center")

    def gate_box(x, y, text, width=0.56, height=0.26, fontsize=8.6):
        rect = patches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=fontsize, ha="center", va="center")

    for layer in range(layers):
        x = x_layer_start + layer * x_layer_gap
        ax.text(x, y_top + 0.38, rf"Layer {layer + 1}", fontsize=10, ha="center", va="center")

        for y, (q_idx, _) in zip(ys, rows):
            if q_idx is None:
                continue
            theta_idx = layer * num_qubits + q_idx
            gate_box(x, y, rf"$R_y(\theta_{{{theta_idx}}})$")

        # In build_ry_chain_ansatz, the CNOT chain is inserted only between RY layers.
        if layer != layers - 1:
            x_cnot = x + 0.82
            valid = [(q_idx, y) for y, (q_idx, _) in zip(ys, rows) if q_idx is not None]
            for (q1, y1), (q2, y2) in zip(valid[:-1], valid[1:]):
                if q2 == q1 + 1:
                    ax.plot([x_cnot, x_cnot], [y1, y2], linewidth=0.9)
                    ax.plot(x_cnot, y1, marker="o", markersize=4)
                    circ = patches.Circle((x_cnot, y2), radius=0.08, fill=False, linewidth=1.0)
                    ax.add_patch(circ)
                    ax.plot([x_cnot - 0.08, x_cnot + 0.08], [y2, y2], linewidth=0.9)
                    ax.plot([x_cnot, x_cnot], [y2 - 0.08, y2 + 0.08], linewidth=0.9)
            ax.text(x_cnot, y_top + 0.38, "CNOT chain", fontsize=9, ha="center", va="center")

    if layers == 1:
        ax.text(
            x_layer_start + 0.95,
            y_top + 0.38,
            r"case setting: $L=1$, no CNOT chain",
            fontsize=9,
            ha="left",
            va="center",
        )

    block_y = y_top - (len(rows) - 1) * row_gap / 2

    gate_box(x_measure, block_y, "measure\nPauli terms\nfor $E,A,C$", width=1.0, height=0.8, fontsize=8.5)
    gate_box(x_decode, block_y, "update\n$\\theta \\leftarrow \\theta + \\Delta\\tau\\dot{\\theta}$", width=1.22, height=0.8, fontsize=8.2)
    gate_box(x_rerank, block_y, "decode samples\n+ LWE-prior\nreranking", width=1.15, height=0.8, fontsize=8.2)

    def arrow(xa, ya, xb, yb):
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya), arrowprops=dict(arrowstyle="->", linewidth=1.1))

    arrow(x_layer_start + (layers - 1) * x_layer_gap + 0.45, block_y, x_measure - 0.55, block_y)
    arrow(x_measure + 0.55, block_y, x_decode - 0.65, block_y)
    arrow(x_decode + 0.65, block_y, x_rerank - 0.65, block_y)

    ax.annotate(
        "",
        xy=(x_layer_start - 0.28, block_y - 0.78),
        xytext=(x_decode, block_y - 0.78),
        arrowprops=dict(arrowstyle="->", linewidth=1.0, connectionstyle="arc3,rad=-0.18"),
    )
    ax.text(
        (x_layer_start + x_decode) / 2,
        block_y - 1.08,
        "repeat for VarQITE parameter-update steps",
        fontsize=9,
        ha="center",
    )

    footer = rf"Case-study size: local dimension $m={m}$, qubits $2m={num_qubits}$, ansatz layers $L={layers}$."
    ax.text(0.5, 0.18, footer, fontsize=9.5, ha="left", va="center")

    ax.set_xlim(0, x_rerank + 0.95)
    ax.set_ylim(-0.05, 5.15)

    save_figure(fig, out_dir / "fig_5_varqite_circuit_schematic")


# =========================================================
# Seed picker from an existing main-experiment run
# =========================================================
def load_source_config(source_run_dir: Path) -> dict[str, Any]:
    cfg_path = source_run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def auto_pick_seed_from_source_run(source_run_dir: Path, n: int | None, m: int | None, q: int | None) -> int:
    raw_path = source_run_dir / "raw_per_seed.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Cannot find raw_per_seed.csv under: {source_run_dir}")

    df = pd.read_csv(raw_path)
    if n is not None and "n" in df.columns:
        df = df[df["n"].astype(int) == int(n)]
    if m is not None and "m" in df.columns:
        df = df[df["m"].astype(int) == int(m)]
    if q is not None and "q" in df.columns:
        df = df[df["q"].astype(int) == int(q)]
    if len(df) == 0:
        raise ValueError("No matching rows found in raw_per_seed.csv for the requested n/m/q.")

    def has_col(c: str) -> bool:
        return c in df.columns

    def bcol(c: str) -> pd.Series:
        return df[c].map(parse_bool)

    # Preference order: most story-friendly first.
    filters: list[tuple[str, pd.Series]] = []
    if has_col("babai_secret_success") and has_col("qite_candidate_best_secret_success") and has_col("classical_secret_success"):
        filters.append((
            "Babai fails, greedy local fails, VarQITE+prior succeeds",
            (~bcol("babai_secret_success")) & (~bcol("classical_secret_success")) & bcol("qite_candidate_best_secret_success"),
        ))
    if has_col("babai_secret_success") and has_col("qite_candidate_best_secret_success"):
        filters.append((
            "Babai fails, VarQITE+prior succeeds",
            (~bcol("babai_secret_success")) & bcol("qite_candidate_best_secret_success"),
        ))
    if has_col("babai_secret_success") and has_col("qite_candidate_any_secret_success"):
        filters.append((
            "Babai fails, VarQITE candidate pool contains a correct candidate",
            (~bcol("babai_secret_success")) & bcol("qite_candidate_any_secret_success"),
        ))
    if has_col("qite_candidate_best_secret_success"):
        filters.append((
            "VarQITE+prior succeeds",
            bcol("qite_candidate_best_secret_success"),
        ))

    for label, mask in filters:
        sub = df[mask].copy()
        if len(sub) > 0:
            sub = sub.sort_values(by=[c for c in ["n", "m", "seed"] if c in sub.columns])
            seed = int(sub.iloc[0]["seed"])
            print(f"[auto-pick] selected seed={seed} ({label})")
            return seed

    seed = int(df.sort_values(by=[c for c in ["n", "m", "seed"] if c in df.columns]).iloc[0]["seed"])
    print(f"[auto-pick] no preferred case found; fallback seed={seed}")
    return seed


# =========================================================
# Core run
# =========================================================
def run_case_study(args: argparse.Namespace) -> Path:
    n = int(args.n)
    q = int(args.q)
    m = int(args.m) if args.m is not None else round_m_from_ratio(n, float(args.c_star), str(args.m_rounding))

    if args.source_run_dir is not None:
        source_dir = (PROJECT_ROOT / args.source_run_dir).resolve() if not Path(args.source_run_dir).is_absolute() else Path(args.source_run_dir)
        source_cfg = load_source_config(source_dir)
        if args.use_source_config and source_cfg:
            # These fields affect instance generation and the main VarQITE/candidate setup.
            for k in [
                "secret_weight_total", "noise_error_rate", "require_nonzero_secret",
                "reduction_method", "delta_lll", "local_values",
                "penalty", "layers", "dtau", "steps", "shots_A_C", "shots_E", "shots_decode",
                "reg_floor", "max_delta_norm", "auto_calibrate",
                "candidate_topk", "candidate_selection", "expected_secret_weight_ratio", "expected_noise_rate",
                "prior_weight_secret_violation", "prior_weight_noise_violation",
                "prior_weight_secret_weight_gap", "prior_weight_noise_weight_gap", "prior_weight_noise_l2",
                "prior_weight_local_energy", "prior_weight_count_log", "candidate_energy_match_tol",
                "decoded_top_states_to_store",
            ]:
                if k in source_cfg and hasattr(args, k):
                    setattr(args, k, source_cfg[k])
            if "q" in source_cfg and args.q is None:
                q = int(source_cfg["q"])
    else:
        source_dir = None

    if args.auto_pick:
        if source_dir is None:
            raise ValueError("--auto-pick requires --source-run-dir.")
        seed = auto_pick_seed_from_source_run(source_dir, n=n, m=m, q=q)
    else:
        seed = int(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"case_n{n}_m{m}_q{q}_seed{seed}_{timestamp}"
    if args.out_dir is None:
        out_dir = PROJECT_ROOT / "output" / "figure4_case_study" / run_name
    else:
        out_dir = (PROJECT_ROOT / args.out_dir / run_name).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("Single-instance VarQITE candidate case study starts.")
    print(f"Artifact root: {PROJECT_ROOT}")
    print(f"Output dir   : {out_dir}")
    print(f"Instance     : n={n}, m={m}, q={q}, seed={seed}")
    print("============================================================\n")

    # Save effective config.
    effective_config = vars(args).copy()
    effective_config.update({"n": n, "m": m, "q": q, "seed": seed})
    json_dump(out_dir / "case_study_config.json", effective_config)

    t_total0 = time.perf_counter()

    # 1. Instance
    inst = safe_generate_instance(n=n, m=m, q=q, seed=seed, args=args)

    # 2. q-ary lattice basis + reduction
    B, _, _ = full_rank_basis_from_hnf(inst.A, q)
    D = reduce_basis(B, method=args.reduction_method, delta=float(args.delta_lll))

    # 3. Babai
    w_babai, _ = babai_nearest_plane(D, inst.c_centered)
    w_babai = normalize_lattice_vector(w_babai)
    residual = np.asarray(inst.c_centered, dtype=int) - w_babai
    initial_residual_energy = float(np.dot(residual, residual))
    delta_zero = np.zeros(D.shape[1], dtype=int)
    s_hat_babai = recover_secret_from_lattice_point(inst.A, w_babai, q)
    babai_success = bool(np.array_equal(s_hat_babai % q, inst.s_mod_q % q))

    # 4. Exact local oracle, optional because 3^m can be expensive.
    delta_exact = None
    exact_best_energy = None
    w_exact = None
    s_hat_exact = None
    exact_success = None
    exact_skipped = False
    local_dim = int(D.shape[1])
    if bool(args.force_exact) or local_dim <= int(args.exact_max_vars):
        print(f"[exact] enumerating local space size={len(args.local_values)}^{local_dim} ...")
        states, energies = build_diagonal_hamiltonian(D=D, residual=residual, values=tuple(int(v) for v in args.local_values))
        idx_best = int(np.argmin(energies))
        delta_exact = np.asarray(states[idx_best], dtype=int)
        exact_best_energy = float(energies[idx_best])
        w_exact = normalize_lattice_vector(w_babai + D @ delta_exact)
        s_hat_exact = recover_secret_from_lattice_point(inst.A, w_exact, q)
        exact_success = bool(np.array_equal(s_hat_exact % q, inst.s_mod_q % q))
    else:
        exact_skipped = True
        print(f"[exact] skipped because local_dim={local_dim} > exact_max_vars={args.exact_max_vars}. Use --force-exact to run it.")

    # 5. Greedy local descent
    t_greedy0 = time.perf_counter()
    greedy_res = greedy_local_from_babai(
        D=D,
        residual=residual,
        values=tuple(int(v) for v in args.local_values),
        max_steps=args.greedy_max_steps,
        tol=float(args.greedy_tol),
    )
    elapsed_greedy = time.perf_counter() - t_greedy0
    delta_greedy = np.asarray(greedy_res.delta_greedy, dtype=int)
    w_greedy = normalize_lattice_vector(w_babai + D @ delta_greedy)
    s_hat_greedy = recover_secret_from_lattice_point(inst.A, w_greedy, q)
    greedy_success = bool(np.array_equal(s_hat_greedy % q, inst.s_mod_q % q))

    # 6. VarQITE
    print("[VarQITE] running local VarQITE ...")
    t_qite0 = time.perf_counter()
    qite_res = qite_optimize_local_pyqpanda3(
        D=D,
        residual=residual,
        penalty=float(args.penalty),
        layers=int(args.layers),
        dtau=float(args.dtau),
        steps=int(args.steps),
        shots_A_C=int(args.shots_A_C),
        shots_E=int(args.shots_E),
        shots_decode=int(args.shots_decode),
        reg_floor=float(args.reg_floor),
        max_delta_norm=float(args.max_delta_norm),
        verbose_every=int(args.verbose_every),
        auto_calibrate=bool(args.auto_calibrate),
        log_steps=bool(args.log_qite_steps),
    )
    elapsed_qite = time.perf_counter() - t_qite0

    # 7. VarQITE candidate generation + LWE-prior reranking
    candidate_cfg = make_candidate_config(args)
    t_cand0 = time.perf_counter()
    qite_cand_res = rerank_varqite_candidates(
        qite_result=qite_res,
        inst=inst,
        D=D,
        w_babai=w_babai,
        residual=residual,
        q=q,
        config=candidate_cfg,
        delta_exact=delta_exact,
        exact_best_energy=exact_best_energy,
    )
    elapsed_qite_rerank = time.perf_counter() - t_cand0

    # 8. Greedy-trajectory candidates + same LWE-prior reranking
    t_classcand0 = time.perf_counter()
    classical_cand_res = rerank_classical_local_candidates(
        inst=inst,
        D=D,
        w_babai=w_babai,
        residual=residual,
        q=q,
        greedy_result=greedy_res,
        rerank_config=candidate_cfg,
        generation_config=make_classical_generation_config(args),
        delta_exact=delta_exact,
        exact_best_energy=exact_best_energy,
    )
    elapsed_classical_rerank = time.perf_counter() - t_classcand0

    qite_best = qite_cand_res.best_candidate
    qite_primary = qite_cand_res.primary_candidate
    classical_best = classical_cand_res.best_candidate

    # =====================================================
    # Save raw data tables
    # =====================================================
    hist_df = pd.DataFrame(qite_res.history)
    if len(hist_df) > 0 and "theta" in hist_df.columns:
        hist_df["theta"] = hist_df["theta"].map(lambda x: json.dumps(to_jsonable(x), ensure_ascii=False))
    dataframe_to_csv(out_dir / "qite_energy_trajectory.csv", hist_df)

    greedy_hist_df = pd.DataFrame(greedy_res.history)
    if len(greedy_hist_df) > 0 and "delta" in greedy_hist_df.columns:
        greedy_hist_df["delta"] = greedy_hist_df["delta"].map(lambda x: json.dumps(to_jsonable(x), ensure_ascii=False))
    dataframe_to_csv(out_dir / "greedy_energy_trajectory.csv", greedy_hist_df)

    qite_df = candidates_to_df(
        qite_cand_res.candidates,
        source="VarQITE candidates",
        selected_delta=[int(v) for v in qite_best["delta"]],
    )
    # add primary flag after serialization-safe reload from original candidates
    if len(qite_df) > 0:
        primary_delta_key = str(tuple(int(v) for v in qite_primary["delta"]))
        qite_df["is_primary"] = qite_df["delta_key"].astype(str).eq(primary_delta_key)
    dataframe_to_csv(out_dir / "varqite_candidate_pool.csv", qite_df)

    classical_df = candidates_to_df(
        classical_cand_res.candidates,
        source="Greedy-traj. candidates",
        selected_delta=[int(v) for v in classical_best["delta"]],
    )
    dataframe_to_csv(out_dir / "greedy_trajectory_candidate_pool.csv", classical_df)

    # Combined plotting dataframe on original numeric values.
    plot_rows: list[dict[str, Any]] = []
    plot_rows.append({
        "plot_group": "Babai",
        "source": "Babai",
        "candidate_id": "babai",
        "delta": vec_to_list(delta_zero),
        "energy": float(initial_residual_energy),
        "lwe_prior_score": np.nan,
        "count": np.nan,
        "secret_success": bool(babai_success),
        "is_selected": False,
        "is_primary": False,
        "is_energy_min": False,
    })
    plot_rows.append({
        "plot_group": "Greedy final",
        "source": "Greedy local descent",
        "candidate_id": "greedy_final",
        "delta": vec_to_list(delta_greedy),
        "energy": float(greedy_res.best_energy),
        "lwe_prior_score": np.nan,
        "count": np.nan,
        "secret_success": bool(greedy_success),
        "is_selected": False,
        "is_primary": False,
        "is_energy_min": False,
    })
    if exact_best_energy is not None:
        plot_rows.append({
            "plot_group": "Exhaustive local enumeration",
            "source": "Exhaustive local enumeration",
            "candidate_id": "exact",
            "delta": vec_to_list(delta_exact),
            "energy": float(exact_best_energy),
            "lwe_prior_score": np.nan,
            "count": np.nan,
            "secret_success": bool(exact_success),
            "is_selected": False,
            "is_primary": False,
            "is_energy_min": True,
        })

    for idx, cand in enumerate(classical_cand_res.candidates):
        plot_rows.append({
            "plot_group": "Greedy-traj. candidates",
            "source": "Greedy-traj. candidates",
            "candidate_id": idx,
            "delta": vec_to_list(cand["delta"]),
            "energy": float(cand["energy"]),
            "lwe_prior_score": float(cand.get("lwe_prior_score", np.nan)),
            "count": int(cand.get("count", 0)),
            "secret_success": cand.get("secret_success"),
            "is_selected": tuple(cand["delta"]) == tuple(classical_best["delta"]),
            "is_primary": False,
            "is_energy_min": False,
        })
    for idx, cand in enumerate(qite_cand_res.candidates):
        plot_rows.append({
            "plot_group": "VarQITE candidates",
            "source": "VarQITE candidates",
            "candidate_id": idx,
            "delta": vec_to_list(cand["delta"]),
            "energy": float(cand["energy"]),
            "lwe_prior_score": float(cand.get("lwe_prior_score", np.nan)),
            "count": int(cand.get("count", 0)),
            "secret_success": cand.get("secret_success"),
            "is_selected": tuple(cand["delta"]) == tuple(qite_best["delta"]),
            "is_primary": tuple(cand["delta"]) == tuple(qite_primary["delta"]),
            "is_energy_min": False,
        })

    plot_df = pd.DataFrame(plot_rows)
    if len(plot_df) > 0:
        for col in ["energy", "lwe_prior_score", "count"]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    dataframe_to_csv(out_dir / "all_candidates_for_plot.csv", plot_df.map(serialize_for_csv))

    # Summary JSON.
    w_true = np.asarray(inst.w_true, dtype=int) if hasattr(inst, "w_true") else None
    summary = {
        "instance": {
            "n": n,
            "m": m,
            "q": q,
            "seed": seed,
            "s_true_centered": vec_to_list(inst.s_centered),
            "s_true_mod_q": vec_to_list(inst.s_mod_q),
            "e_true": vec_to_list(inst.e),
            "c_centered": vec_to_list(inst.c_centered),
        },
        "baseline_results": {
            "babai_secret_success": babai_success,
            "initial_residual_energy": initial_residual_energy,
            "greedy_secret_success": greedy_success,
            "greedy_best_energy": float(greedy_res.best_energy),
            "exact_skipped": exact_skipped,
            "exact_secret_success": exact_success,
            "exact_best_energy": exact_best_energy,
        },
        "varqite_results": {
            "primary_delta": qite_primary.get("delta"),
            "primary_energy": qite_primary.get("energy"),
            "primary_secret_success": qite_primary.get("secret_success"),
            "candidate_best_delta": qite_best.get("delta"),
            "candidate_best_energy": qite_best.get("energy"),
            "candidate_best_lwe_prior_score": qite_best.get("lwe_prior_score"),
            "candidate_best_count": qite_best.get("count"),
            "candidate_best_secret_success": qite_best.get("secret_success"),
            "candidate_num_unique": qite_cand_res.candidate_num_unique,
            "candidate_any_secret_success": qite_cand_res.any_secret_success,
        },
        "greedy_trajectory_rerank_results": {
            "candidate_best_delta": classical_best.get("delta"),
            "candidate_best_energy": classical_best.get("energy"),
            "candidate_best_lwe_prior_score": classical_best.get("lwe_prior_score"),
            "candidate_best_secret_success": classical_best.get("secret_success"),
            "candidate_topk_used": classical_cand_res.candidate_topk_used,
            "candidate_any_secret_success": classical_cand_res.any_secret_success,
        },
        "timing_sec": {
            "greedy": elapsed_greedy,
            "varqite": elapsed_qite,
            "varqite_rerank": elapsed_qite_rerank,
            "greedy_trajectory_rerank": elapsed_classical_rerank,
            "total": time.perf_counter() - t_total0,
        },
    }
    if w_true is not None:
        summary["geometry"] = {
            "dist2_true_babai": int(np.sum((w_babai - w_true) ** 2)),
            "dist2_true_greedy": int(np.sum((w_greedy - w_true) ** 2)),
            "dist2_true_varqite_best": int(qite_best.get("dist2_true")) if qite_best.get("dist2_true") is not None else None,
            "dist2_true_greedy_trajectory_best": int(classical_best.get("dist2_true")) if classical_best.get("dist2_true") is not None else None,
        }
    json_dump(out_dir / "case_summary.json", summary)

    # =====================================================
    # Merged mechanism figure: former Figures 1 and 2/4 in a 2 x 2 layout
    #   (a) VarQITE energy trajectory
    #   (b) 3D real-parameter energy surface
    #   (c) Candidate-pool local-energy distribution
    #   (d) Top-20 VarQITE candidates after LWE-prior reranking
    #
    # The subfigure titles intentionally contain ONLY (a), (b), (c), (d).
    # Descriptive wording should be placed in the paper caption instead.
    # =====================================================
    qite_plot_df = plot_df[plot_df["plot_group"] == "VarQITE candidates"].copy()

    if len(hist_df) > 0 and len(qite_plot_df) > 0:
        fig = plt.figure(figsize=FIG_MECHANISM_4PANEL_SIZE_IN)
        gs = fig.add_gridspec(
            2,
            2,
            left=0.075,
            right=0.962,
            bottom=0.090,
            top=0.965,
            wspace=0.25,
            hspace=0.25,
        )

        ax_traj = fig.add_subplot(gs[0, 0])
        ax_surface = fig.add_subplot(gs[0, 1], projection="3d")
        ax_dist = fig.add_subplot(gs[1, 0])
        ax_rank = fig.add_subplot(gs[1, 1])

        # -------------------------------------------------
        # Geometry alignment for the right column
        # -------------------------------------------------
        # Use panel (a) as the vertical reference for panel (b), and use
        # panel (d) as the horizontal reference for panel (b). This gives:
        #   - (a) and (b): exactly the same y0 and height
        #   - (b) and (d): exactly the same x0 and width
        # The right column is slightly narrowed to leave enough room for
        # the colorbar in (b) and the secondary y-axis labels in (d).
        box_a = ax_traj.get_position()
        box_d0 = ax_rank.get_position()

        right_col_x0 = box_d0.x0 - 0.004
        surface_w = box_d0.width * 0.87
        rank_w = box_d0.width * 0.91

        # (d) 稍微加宽
        ax_rank.set_position([
            right_col_x0,
            box_d0.y0,
            rank_w,
            box_d0.height,
        ])

        # (b) 稍微左移，宽度保持较紧凑
        ax_surface.set_position([
            right_col_x0 - 0.05,
            box_a.y0,
            surface_w,
            box_a.height,
        ])

        # -------------------------------------------------
        # (a) VarQITE energy trajectory
        # -------------------------------------------------
        x = pd.to_numeric(hist_df["step"], errors="coerce")

        if "E_before" in hist_df.columns:
            ax_traj.plot(
                x,
                pd.to_numeric(hist_df["E_before"], errors="coerce"),
                marker="o",
                markersize=4.0,
                label="Energy before parameters update",
            )

        if "E_after" in hist_df.columns:
            ax_traj.plot(
                x,
                pd.to_numeric(hist_df["E_after"], errors="coerce"),
                marker="s",
                markersize=4.0,
                label="Energy after parameters update",
            )

        ax_traj.set_xlabel(
            "VarQITE parameter update step",
            fontsize=SCIS_FONT_SIZE,
            labelpad=4,
        )
        ax_traj.set_ylabel(
            "Measured encoded Hamiltonian energy",
            fontsize=SCIS_FONT_SIZE,
        )
        ax_traj.set_title("(a)", fontsize=SCIS_FONT_SIZE, fontweight="normal", pad=2)
        apply_scis_axis_style(ax_traj, grid=True, grid_axis="both")
        ax_traj.legend(
            frameon=False,
            fontsize=SCIS_FONT_SIZE,
            loc="upper right",
            borderaxespad=0.2,
            labelspacing=0.3,
            handlelength=1.8,
            handletextpad=0.5,
        )

        # -------------------------------------------------
        # (b) 3D energy surface over two real parameters
        # -------------------------------------------------
        surf, param_i, param_j = plot_varqite_real_parameter_energy_surface(
            ax_surface,
            qite_res=qite_res,
            D=D,
            residual=residual,
            args=args,
        )
        # Override the descriptive title created inside the helper.
        ax_surface.set_title("(b)", fontsize=SCIS_FONT_SIZE, fontweight="normal", pad=1)

        # The 3D panel position has already been aligned with (a) and (d)
        # above. Put the colorbar farther to the right in its own axes so it
        # never overlaps with the 3D z-axis labels/ticks.
        if surf is not None:
            surface_box = ax_surface.get_position()
            cax = fig.add_axes([
                surface_box.x1 + 0.07,
                surface_box.y0 + 0.12 * surface_box.height,
                0.012,
                0.72 * surface_box.height,
            ])
            cbar = fig.colorbar(surf, cax=cax)
            cbar.set_label(
                "Measured encoded Hamiltonian energy",
                fontsize=SCIS_FONT_SIZE,
                labelpad=4,
            )
            cbar.ax.tick_params(
                labelsize=SCIS_FONT_SIZE,
                width=SCIS_LINEWIDTH_PT,
                length=3,
            )
            cbar.outline.set_linewidth(SCIS_LINEWIDTH_PT)

        # -------------------------------------------------
        # (c) Candidate-pool local-energy distribution
        # -------------------------------------------------
        categories = [
            "Babai",
            "Exhaustive local enumeration",
            "Greedy-traj. candidates",
            "VarQITE candidates",
        ]
        categories = [c for c in categories if c in set(plot_df["plot_group"])]
        x_map = {name: i for i, name in enumerate(categories)}
        rng = np.random.default_rng(2026)

        color_map = {
            "Babai": "tab:blue",
            "Exhaustive local enumeration": "tab:purple",
            "Greedy-traj. candidates": "tab:green",
            "VarQITE candidates": "tab:red",
        }

        legend_label_map = {
            "Babai": "Babai",
            "Exhaustive local enumeration": "Exhaustive local enumeration",
            "VarQITE candidates": "VarQITE candidates",
        }
        for name in categories:
            sub = plot_df[plot_df["plot_group"] == name].copy()
            y = pd.to_numeric(sub["energy"], errors="coerce")
            x0 = x_map[name]
            jitter = rng.uniform(-0.07, 0.07, size=len(sub)) if len(sub) > 1 else np.zeros(len(sub))
            ax_dist.scatter(
                np.full(len(sub), x0) + jitter,
                y,
                alpha=0.75,
                s=28,
                linewidths=SCIS_MARKER_EDGEWIDTH_PT,
                color=color_map.get(name, None),
                label=legend_label_map.get(name, "_nolegend_"),
            )

        selected_style = {
            "Selected Greedy-traj.+rerank candidate": "tab:brown",
            "Selected VarQITE+rerank candidate": "tab:pink",
        }
        for label, source_name, cand in [
            ("Selected Greedy-traj.+rerank candidate", "Greedy-traj. candidates", classical_best),
            ("Selected VarQITE+rerank candidate", "VarQITE candidates", qite_best),
        ]:
            if source_name in x_map:
                ax_dist.scatter(
                    [x_map[source_name]],
                    [float(cand["energy"])],
                    marker="*",
                    s=150,
                    linewidths=SCIS_MARKER_EDGEWIDTH_PT,
                    color=selected_style.get(label, None),
                    label=label,
                    zorder=5,
                )

        compact_tick_labels = {
            "Babai": "Babai",
            "Exhaustive local enumeration": "Exhaustive\nlocal enum.",
            "Greedy-traj. candidates": "Greedy-traj.\ncandidates",
            "VarQITE candidates": "VarQITE\ncandidates",
        }
        ax_dist.set_xticks(list(x_map.values()))
        ax_dist.set_xticklabels(
            [compact_tick_labels.get(name, name) for name in x_map.keys()],
            rotation=0,
            ha="center",
            fontsize=SCIS_FONT_SIZE,
            linespacing=0.95,
        )
        ax_dist.set_ylabel(
            r"Local decoding energy $\|r-D\delta\|^2$",
            fontsize=SCIS_FONT_SIZE,
        )
        ax_dist.set_title("(c)", fontsize=SCIS_FONT_SIZE, fontweight="normal", pad=2)
        apply_scis_axis_style(ax_dist, grid=True, grid_axis="y")
        ax_dist.legend(
            frameon=False,
            fontsize=SCIS_FONT_SIZE,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            borderaxespad=0.0,
            labelspacing=0.25,
            handletextpad=0.4,
        )

        # -------------------------------------------------
        # (d) Candidate ranking after LWE-prior reranking
        # -------------------------------------------------
        rank_df = qite_plot_df.copy()
        rank_df["energy"] = pd.to_numeric(rank_df["energy"], errors="coerce")
        rank_df["lwe_prior_score"] = pd.to_numeric(rank_df["lwe_prior_score"], errors="coerce")
        rank_df["count"] = pd.to_numeric(rank_df["count"], errors="coerce")
        rank_df = rank_df.sort_values(
            ["lwe_prior_score", "energy", "count"],
            ascending=[True, True, False],
        ).head(min(20, len(rank_df)))
        rank_df = rank_df.reset_index(drop=True)
        rank_df["rank"] = np.arange(1, len(rank_df) + 1)

        rank_x = rank_df["rank"].to_numpy(dtype=float)

        ax_rank.bar(
            rank_x,
            rank_df["energy"].to_numpy(dtype=float),
            width=0.8,
            align="center",
            alpha=0.7,
            linewidth=SCIS_LINEWIDTH_PT,
            edgecolor="black",
            label="Local decoding energy",
        )
        ax_rank.set_xlabel(
            "Candidate ranking after reranking",
            fontsize=SCIS_FONT_SIZE,
            labelpad=3,
        )
        ax_rank.set_ylabel(
            r"Local decoding energy $\|r-D\delta\|^2$",
            fontsize=SCIS_FONT_SIZE,
        )
        tick_candidates = [v for v in [1, 5, 10, 15, 20] if v <= len(rank_df)]
        ax_rank.set_xticks(tick_candidates)
        ax_rank.set_title("(d)", fontsize=SCIS_FONT_SIZE, fontweight="normal", pad=2)
        apply_scis_axis_style(ax_rank, grid=True, grid_axis="y")

        ax_rank_r = ax_rank.twinx()
        # Explicitly match the secondary axis position to the primary axis.
        # This prevents any visual horizontal offset after manual resizing.
        ax_rank_r.set_position(ax_rank.get_position())
        ax_rank_r.plot(
            rank_x,
            rank_df["lwe_prior_score"].to_numpy(dtype=float),
            marker="o",
            markersize=4.0,
            linewidth=SCIS_LINEWIDTH_PT,
            label="Reranking score",
        )

        # Force both y-axes to use exactly the same x coordinates and x range.
        # This keeps every reranking-score marker centered on its bar even
        # after the panel width is changed manually.
        x_left = 0.5
        x_right = len(rank_df) + 0.5
        ax_rank.set_xlim(x_left, x_right)
        ax_rank_r.set_xlim(x_left, x_right)
        ax_rank.margins(x=0)
        ax_rank_r.margins(x=0)
        ax_rank_r.set_ylabel(
            "Reranking score",
            fontsize=SCIS_FONT_SIZE,
        )
        ax_rank_r.tick_params(
            axis="y",
            labelsize=SCIS_FONT_SIZE,
            width=SCIS_LINEWIDTH_PT,
            length=3,
            pad=1,
        )
        for spine in ax_rank_r.spines.values():
            spine.set_linewidth(SCIS_LINEWIDTH_PT)

        def delta_to_key_for_plot(x: Any) -> str:
            if isinstance(x, str):
                try:
                    parsed = json.loads(x) if x.startswith("[") else ast.literal_eval(x)
                    return str(tuple(int(v) for v in parsed))
                except Exception:
                    return str(x)
            return str(tuple(int(v) for v in np.asarray(x).reshape(-1)))

        selected_delta_key = str(tuple(int(v) for v in qite_best["delta"]))
        selected_rows = rank_df[
            rank_df["delta"].map(delta_to_key_for_plot) == selected_delta_key
        ]
        if len(selected_rows) > 0:
            r = int(selected_rows.iloc[0]["rank"])
            ax_rank.axvline(
                r,
                linestyle="--",
                linewidth=SCIS_LINEWIDTH_PT,
                label="Selected candidate",
            )

        h1, l1 = ax_rank.get_legend_handles_labels()
        h2, l2 = ax_rank_r.get_legend_handles_labels()
        ax_rank.legend(
            h1 + h2,
            l1 + l2,
            frameon=False,
            fontsize=SCIS_FONT_SIZE,
            loc="upper left",
            bbox_to_anchor=(0.03, 0.98),
            borderaxespad=0.0,
            labelspacing=0.25,
            handletextpad=0.4,
        )

        save_figure(
            fig,
            out_dir / "figure4_mechanism_analysis",
            use_tight_layout=False,
        )

    # =====================================================
    # Figure 3: VarQITE candidates: energy vs LWE-prior score
    # =====================================================
    if len(qite_plot_df) > 0:
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        x = pd.to_numeric(qite_plot_df["energy"], errors="coerce")
        y = pd.to_numeric(qite_plot_df["lwe_prior_score"], errors="coerce")
        counts = pd.to_numeric(qite_plot_df["count"], errors="coerce").fillna(0).to_numpy()
        sizes = 40 + 35 * np.log1p(np.maximum(counts, 0))
        ax.scatter(x, y, s=sizes, alpha=0.65, label="VarQITE decoded candidates")

        # Correct candidates, if any, shown with hollow markers.
        if "secret_success" in qite_plot_df.columns:
            correct_mask = qite_plot_df["secret_success"].map(parse_bool)
            if correct_mask.any():
                ax.scatter(
                    pd.to_numeric(qite_plot_df.loc[correct_mask, "energy"], errors="coerce"),
                    pd.to_numeric(qite_plot_df.loc[correct_mask, "lwe_prior_score"], errors="coerce"),
                    s=sizes[correct_mask.to_numpy()],
                    facecolors="none",
                    linewidths=1.5,
                    label="Secret-correct candidate",
                )

        ax.scatter([float(qite_best["energy"])], [float(qite_best["lwe_prior_score"])], marker="*", s=240, label="Selected by LWE-prior", zorder=5)
        energy_min_idx = x.idxmin()
        ax.scatter([x.loc[energy_min_idx]], [y.loc[energy_min_idx]], marker="X", s=130, label="Lowest local energy", zorder=4)
        if qite_primary is not None:
            ax.scatter([float(qite_primary["energy"])], [float(qite_primary["lwe_prior_score"])], marker="P", s=130, label="Primary decoded output", zorder=4)

        ax.set_xlabel(r"Local BDD energy $\|r-D\delta\|^2$")
        ax.set_ylabel("LWE-prior score (lower is better)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        save_figure(fig, out_dir / "fig_3_varqite_energy_prior_scatter")

    # =====================================================
    # Figure 5: VarQITE circuit schematic
    # =====================================================
    draw_varqite_circuit_schematic(
        out_dir=out_dir,
        m=int(D.shape[1]),
        layers=int(args.layers),
    )

    print("\n============================================================")
    print("Done. Outputs saved to:")
    print(out_dir)
    print("Main files:")
    for p in [
        "case_summary.json",
        "qite_energy_trajectory.csv",
        "varqite_candidate_pool.csv",
        "greedy_trajectory_candidate_pool.csv",
        "all_candidates_for_plot.csv",
        "figure4_mechanism_analysis.png",
        "fig_3_varqite_energy_prior_scatter.png",
        "fig_5_varqite_circuit_schematic.png",
    ]:
        print("  -", out_dir / p)
    print("============================================================\n")

    return out_dir


# =========================================================
# CLI
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-instance VarQITE candidate + LWE-prior case-study plotting script.")

    # Instance and seed selection
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--m", type=int, default=None, help="If omitted, m=ceil(c_star*n) by default.")
    p.add_argument("--c-star", dest="c_star", type=float, default=1.325)
    p.add_argument("--m-rounding", dest="m_rounding", type=str, default="ceil", choices=["ceil", "round", "floor"])
    p.add_argument("--q", type=int, default=17)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--source-run-dir", type=str, default=None, help="Existing main-experiment run dir for auto-picking a representative seed.")
    p.add_argument("--auto-pick", action="store_true", help="Pick a representative seed from --source-run-dir.")
    p.add_argument("--use-source-config", action="store_true", help="Load compatible parameters from source run config.json.")
    p.add_argument("--out-dir", type=str, default=None, help="Output root. A timestamped run directory is created inside it. Default: output/figure4_case_study/")

    # LWE distribution
    p.add_argument("--secret-weight-total", dest="secret_weight_total", type=float, default=0.5)
    p.add_argument("--noise-error-rate", dest="noise_error_rate", type=float, default=0.25)
    p.add_argument("--secret-mode", dest="secret_mode", type=str, default=None)
    p.add_argument("--noise-mode", dest="noise_mode", type=str, default=None)
    p.add_argument("--require-nonzero-secret", dest="require_nonzero_secret", action="store_true", default=True)
    p.add_argument("--allow-zero-secret", dest="require_nonzero_secret", action="store_false")

    # Lattice and local model
    p.add_argument("--reduction-method", dest="reduction_method", type=str, default="lll")
    p.add_argument("--delta-lll", dest="delta_lll", type=float, default=0.75)
    p.add_argument("--local-values", dest="local_values", type=int, nargs="+", default=[-1, 0, 1])
    p.add_argument("--exact-max-vars", dest="exact_max_vars", type=int, default=12, help="Skip exact local oracle if local dimension exceeds this value.")
    p.add_argument("--force-exact", dest="force_exact", action="store_true", help="Force exact local enumeration even when local dimension is large.")

    # Greedy local descent
    p.add_argument("--greedy-max-steps", dest="greedy_max_steps", type=int, default=None)
    p.add_argument("--greedy-tol", dest="greedy_tol", type=float, default=1e-12)

    # VarQITE
    p.add_argument("--penalty", type=float, default=10.0)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dtau", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--shots-A-C", dest="shots_A_C", type=int, default=100)
    p.add_argument("--shots-E", dest="shots_E", type=int, default=100)
    p.add_argument("--shots-decode", dest="shots_decode", type=int, default=500)
    p.add_argument("--reg-floor", dest="reg_floor", type=float, default=0.1)
    p.add_argument("--max-delta-norm", dest="max_delta_norm", type=float, default=0.05)
    p.add_argument("--verbose-every", dest="verbose_every", type=int, default=1000)
    p.add_argument("--auto-calibrate", dest="auto_calibrate", action="store_true", default=False)
    p.add_argument("--log-qite-steps", dest="log_qite_steps", action="store_true", default=False)
    # Fig. 1(b): smooth 3D energy surface over two REAL variational parameters.
    p.add_argument("--real-surface-grid-size", dest="real_surface_grid_size", type=int, default=25, help="Grid size per axis for the smooth real-parameter 3D energy surface in Fig. 1(b).")
    p.add_argument("--real-surface-padding", dest="real_surface_padding", type=float, default=0.35, help="Padding ratio around the actual VarQITE trajectory range in the selected real-parameter axes.")
    p.add_argument("--real-surface-shots", dest="real_surface_shots", type=int, default=120, help="Shots per grid point for the real-parameter 3D energy surface in Fig. 1(b).")
    p.add_argument("--real-surface-smooth-passes", dest="real_surface_smooth_passes", type=int, default=2, help="Number of simple smoothing passes applied to the real-parameter energy surface.")

    # Backward-compatible aliases for previous trajectory-PCA commands.
    # They are kept hidden; the new Fig. 1(b) uses --real-surface-* options.
    p.add_argument("--trajectory-surface-grid-size", dest="trajectory_surface_grid_size", type=int, default=25, help=argparse.SUPPRESS)
    p.add_argument("--trajectory-surface-padding", dest="trajectory_surface_padding", type=float, default=0.28, help=argparse.SUPPRESS)
    p.add_argument("--trajectory-surface-shots", dest="trajectory_surface_shots", type=int, default=120, help=argparse.SUPPRESS)
    p.add_argument("--trajectory-surface-smooth-passes", dest="trajectory_surface_smooth_passes", type=int, default=2, help=argparse.SUPPRESS)

    # Backward-compatible aliases kept for old commands; they are no longer used to define random samples.
    p.add_argument("--landscape-samples", dest="landscape_samples", type=int, default=300, help=argparse.SUPPRESS)
    p.add_argument("--landscape-sigma", dest="landscape_sigma", type=float, default=0.18, help=argparse.SUPPRESS)
    p.add_argument("--landscape-shots", dest="landscape_shots", type=int, default=100, help=argparse.SUPPRESS)
    p.add_argument("--landscape-random-seed", dest="landscape_random_seed", type=int, default=2027, help=argparse.SUPPRESS)
    p.add_argument("--landscape-sampling", dest="landscape_sampling", type=str, default="movement", choices=["movement", "isotropic"], help=argparse.SUPPRESS)
    p.add_argument("--no-landscape-contour", dest="landscape_contour", action="store_false", default=True, help=argparse.SUPPRESS)

    # Candidate + prior reranking
    p.add_argument("--candidate-topk", dest="candidate_topk", type=int, default=20)
    p.add_argument("--candidate-selection", dest="candidate_selection", type=str, default="energy_lwe_prior")
    p.add_argument("--expected-secret-weight-ratio", dest="expected_secret_weight_ratio", type=float, default=0.5)
    p.add_argument("--expected-noise-rate", dest="expected_noise_rate", type=float, default=0.25)
    p.add_argument("--prior-weight-secret-violation", dest="prior_weight_secret_violation", type=float, default=1000.0)
    p.add_argument("--prior-weight-noise-violation", dest="prior_weight_noise_violation", type=float, default=1000.0)
    p.add_argument("--prior-weight-secret-weight-gap", dest="prior_weight_secret_weight_gap", type=float, default=20.0)
    p.add_argument("--prior-weight-noise-weight-gap", dest="prior_weight_noise_weight_gap", type=float, default=10.0)
    p.add_argument("--prior-weight-noise-l2", dest="prior_weight_noise_l2", type=float, default=1.0)
    p.add_argument("--prior-weight-local-energy", dest="prior_weight_local_energy", type=float, default=0.05)
    p.add_argument("--prior-weight-count-log", dest="prior_weight_count_log", type=float, default=0.1)
    p.add_argument("--candidate-energy-match-tol", dest="candidate_energy_match_tol", type=float, default=1e-9)
    p.add_argument("--decoded-top-states-to-store", dest="decoded_top_states_to_store", type=int, default=10)

    # Classical candidate pool
    p.add_argument("--include-single-coordinate-neighbors", dest="include_single_coordinate_neighbors", action="store_true", default=False)
    p.add_argument("--classical-preselection-mode", dest="classical_preselection_mode", type=str, default="energy_then_l1")
    p.add_argument("--classical-preview-states-to-store", dest="classical_preview_states_to_store", type=int, default=10)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_case_study(args)


if __name__ == "__main__":
    main()
