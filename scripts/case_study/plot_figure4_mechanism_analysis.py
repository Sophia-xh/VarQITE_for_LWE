# -*- coding: utf-8 -*-
"""
plot_figure4_mechanism_analysis.py
=================================

Regenerate the four-panel mechanism-analysis figure (Figure 4 of the paper)
directly from the archived case-study data, without recomputing anything.

This script performs NO computation of its own. It does not regenerate the LWE
instance, and it does not run Babai, exhaustive local enumeration, greedy local
descent, VarQITE, or any quantum-circuit simulation. It only reads the archived
files written by the original case-study run and redraws them.

Input files, all read from --data-dir (default: data/figure4_mechanism_analysis):

    case_summary.json            instance and per-method outcomes
    case_study_config.json       the resolved configuration of the archived run
    qite_energy_trajectory.csv   per-update energies and parameter vectors
    all_candidates_for_plot.csv  combined candidate table for panels (c) and (d)

The plotting style is taken verbatim from the end-to-end case-study script,
    scripts/case_study/run_figure4_case_study.py,
which is used here purely as a style reference: figure size, gridspec geometry
and manual panel alignment, fonts, font sizes, line widths, marker sizes and
shapes, colours, axis labels, tick locators, axis ranges, legend placement,
3D view angle, and the deterministic jitter seed are all reproduced unchanged.

Panel coverage
--------------
(a), (c), (d) are reproduced exactly. Every plotted value comes from the
archived CSV files, so these three panels are numerically identical to the
archived figure.

(b) is reproduced from the archived VarQITE parameter trajectory: the same two
variational parameters are selected by the same rule (largest movement range
along the trajectory), giving the same theta_10 / theta_15 axes, the same padded
axis ranges, the same smoothed 3D path, the same initial/final markers, the same
view angle and the same labels.

The underlying 25 x 25 energy surface mesh of panel (b) is NOT part of the
archive. In the original run it was measured with 625 shot-based circuit
evaluations, and those grid values were never written to disk. Recomputing them
would require exactly the quantum-circuit simulation this script must avoid, and
reconstructing them by any other means would fabricate data that is not in the
archive. Therefore:

  * by default, panel (b) is drawn as the archived trajectory over the same
    axes, without the surface mesh;
  * if a surface grid is supplied with --surface-csv (columns theta_i, theta_j,
    energy on a regular grid), it is rendered exactly as in the original script,
    including the smoothing passes and the colorbar.

This limitation is recorded in README.md and ARTIFACT_AUDIT.md. To obtain a figure that also
contains the measured surface, re-run the original end-to-end case-study script,
which repeats the simulation.

Usage
-----
    python scripts/case_study/plot_figure4_mechanism_analysis.py

    # only some formats, written somewhere else
    python scripts/case_study/plot_figure4_mechanism_analysis.py \
        --out-dir <dir> --formats png pdf
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.interpolate import splprep, splev
except Exception:  # pragma: no cover
    splprep = None
    splev = None


# =========================================================
# Style constants, copied from the original case-study script
# =========================================================
SCIS_FONT_SIZE = 8
SCIS_PNG_DPI = 600
SCIS_LINEWIDTH_PT = 0.6
SCIS_MARKER_EDGEWIDTH_PT = 0.6

SCIS_TEXT_WIDTH_MM = 176.0
SCIS_TEXT_WIDTH_IN = SCIS_TEXT_WIDTH_MM / 25.4

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = "data/figure4_mechanism_analysis"
DEFAULT_OUT_DIR = "output/figure4_mechanism_analysis"
DEFAULT_BASENAME = "figure4_mechanism_analysis"


# =========================================================
# Helpers, copied from the original case-study script
# =========================================================
def apply_scis_axis_style(ax: plt.Axes, *, grid: bool = True, grid_axis: str = "both") -> None:
    ax.tick_params(axis="both", labelsize=SCIS_FONT_SIZE, width=SCIS_LINEWIDTH_PT, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(SCIS_LINEWIDTH_PT)
    if grid:
        ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=SCIS_LINEWIDTH_PT)


def _smooth_grid_2d_numpy(Z: np.ndarray, passes: int = 2) -> np.ndarray:
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


def _smooth_path_for_plot(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, n_points: int = 240
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 1e-12:
        return x, y, z
    u_f = np.linspace(0.0, t[-1], n_points)
    return np.interp(u_f, t, x), np.interp(u_f, t, y), np.interp(u_f, t, z)


def _select_two_real_parameter_axes_from_trajectory(theta_traj: np.ndarray) -> tuple[int, int]:
    """Two parameters with the largest movement range along the trajectory."""
    theta_traj = np.asarray(theta_traj, dtype=float)
    if theta_traj.ndim != 2 or theta_traj.shape[1] < 2:
        raise ValueError("Need at least two real variational parameters to draw the 3D panel.")
    ranges = np.nanmax(theta_traj, axis=0) - np.nanmin(theta_traj, axis=0)
    order = np.argsort(ranges)[::-1]
    i, j = int(order[0]), int(order[1])
    return min(i, j), max(i, j)


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes"}
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return bool(x)
    return False


def parse_vector(x: Any) -> list:
    if isinstance(x, str):
        s = x.strip()
        try:
            return json.loads(s) if s.startswith("[") else list(ast.literal_eval(s))
        except Exception:
            return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return []


def delta_to_key_for_plot(x: Any) -> str:
    if isinstance(x, str):
        try:
            parsed = json.loads(x) if x.strip().startswith("[") else ast.literal_eval(x)
            return str(tuple(int(v) for v in parsed))
        except Exception:
            return str(x)
    return str(tuple(int(v) for v in np.asarray(x).reshape(-1)))


# =========================================================
# Archived-data loading
# =========================================================
def resolve_dir(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def load_archived_data(data_dir: Path) -> dict[str, Any]:
    required = [
        "case_summary.json",
        "case_study_config.json",
        "qite_energy_trajectory.csv",
        "all_candidates_for_plot.csv",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing archived file(s) in {data_dir}: {', '.join(missing)}"
        )

    with open(data_dir / "case_summary.json", "r", encoding="utf-8-sig") as f:
        summary = json.load(f)
    with open(data_dir / "case_study_config.json", "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    hist_df = pd.read_csv(data_dir / "qite_energy_trajectory.csv", encoding="utf-8-sig")
    plot_df = pd.read_csv(data_dir / "all_candidates_for_plot.csv", encoding="utf-8-sig")

    return {
        "summary": summary,
        "config": config,
        "hist_df": hist_df,
        "plot_df": plot_df,
    }


def load_surface_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optional regular-grid surface for panel (b).

    Expected columns: theta_i, theta_j, energy. The rows must form a complete
    regular grid; the ordering does not matter.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    for want in ("theta_i", "theta_j", "energy"):
        if want not in cols:
            raise ValueError(
                f"{path} must contain columns theta_i, theta_j, energy; found {list(df.columns)}"
            )
    xi = pd.to_numeric(df[cols["theta_i"]], errors="coerce").to_numpy(dtype=float)
    yj = pd.to_numeric(df[cols["theta_j"]], errors="coerce").to_numpy(dtype=float)
    zz = pd.to_numeric(df[cols["energy"]], errors="coerce").to_numpy(dtype=float)

    xs = np.unique(xi)
    ys = np.unique(yj)
    if len(xs) * len(ys) != len(df):
        raise ValueError(
            f"{path} does not describe a complete regular grid "
            f"({len(xs)} x {len(ys)} != {len(df)} rows)."
        )
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.full_like(XX, np.nan, dtype=float)
    ix = np.searchsorted(xs, xi)
    iy = np.searchsorted(ys, yj)
    ZZ[iy, ix] = zz
    if not np.all(np.isfinite(ZZ)):
        raise ValueError(f"{path} has missing grid points.")
    return XX, YY, ZZ


# =========================================================
# Trajectory reconstruction for panels (a) and (b)
# =========================================================
def build_trajectory(hist_df: pd.DataFrame) -> dict[str, Any]:
    """
    Rebuild the archived VarQITE parameter path and its measured energies.

    The archived CSV stores, for each update step, the energy before the update,
    the energy after the update, and the post-update parameter vector. The
    original script plotted the path through the post-update parameter vectors
    with their corresponding post-update energies, preceded by the initial
    parameter vector. The initial vector itself is not stored in the archive
    (it is derived from the instance), so the reconstructed path starts at the
    first post-update point.
    """
    theta_rows = [np.asarray(parse_vector(s), dtype=float) for s in hist_df["theta"]]
    widths = {len(t) for t in theta_rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent parameter-vector lengths in the archive: {sorted(widths)}")

    theta_traj = np.vstack(theta_rows)
    energy_traj = pd.to_numeric(hist_df["E_after"], errors="coerce").to_numpy(dtype=float)

    param_i, param_j = _select_two_real_parameter_axes_from_trajectory(theta_traj)
    return {
        "theta_traj": theta_traj,
        "energy_traj": energy_traj,
        "param_i": param_i,
        "param_j": param_j,
    }


# =========================================================
# The four panels
# =========================================================
def draw_panel_a(ax: plt.Axes, hist_df: pd.DataFrame) -> None:
    x = pd.to_numeric(hist_df["step"], errors="coerce")

    if "E_before" in hist_df.columns:
        ax.plot(
            x,
            pd.to_numeric(hist_df["E_before"], errors="coerce"),
            marker="o",
            markersize=4.0,
            label="Energy before parameters update",
        )

    if "E_after" in hist_df.columns:
        ax.plot(
            x,
            pd.to_numeric(hist_df["E_after"], errors="coerce"),
            marker="s",
            markersize=4.0,
            label="Energy after parameters update",
        )

    ax.set_xlabel("VarQITE parameter update step", fontsize=SCIS_FONT_SIZE, labelpad=4)
    ax.set_ylabel("Measured encoded Hamiltonian energy", fontsize=SCIS_FONT_SIZE)
    ax.set_title("(a)", fontsize=SCIS_FONT_SIZE, fontweight="normal", pad=2)
    apply_scis_axis_style(ax, grid=True, grid_axis="both")
    ax.legend(
        frameon=False,
        fontsize=SCIS_FONT_SIZE,
        loc="upper right",
        borderaxespad=0.2,
        labelspacing=0.3,
        handlelength=1.8,
        handletextpad=0.5,
    )


def draw_panel_b(
    ax,
    traj: dict[str, Any],
    config: dict[str, Any],
    surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
):
    """Panel (b): archived VarQITE path over two real variational parameters."""
    theta_traj = traj["theta_traj"]
    energy_traj = traj["energy_traj"]
    param_i = traj["param_i"]
    param_j = traj["param_j"]

    traj_x = theta_traj[:, param_i]
    traj_y = theta_traj[:, param_j]

    grid_size = int(config.get("real_surface_grid_size", 25) or 25)
    grid_size = max(9, grid_size)
    padding_ratio = float(config.get("real_surface_padding", 0.35))
    smooth_passes = int(config.get("real_surface_smooth_passes", 2) or 0)

    x_min, x_max = float(np.min(traj_x)), float(np.max(traj_x))
    y_min, y_max = float(np.min(traj_y)), float(np.max(traj_y))
    x_span = max(x_max - x_min, 0.18)
    y_span = max(y_max - y_min, 0.18)
    x_pad = padding_ratio * x_span
    y_pad = padding_ratio * y_span

    surf = None
    if surface is not None:
        XX, YY, ZZ = surface
        ZZ_plot = _smooth_grid_2d_numpy(ZZ, passes=smooth_passes)
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
        z_span = float(np.nanmax(ZZ_plot) - np.nanmin(ZZ_plot))
    else:
        z_span = float(np.nanmax(energy_traj) - np.nanmin(energy_traj))

    path_x_s, path_y_s, path_z_s = _smooth_path_for_plot(
        traj_x, traj_y, energy_traj, n_points=240
    )
    ax.plot(
        path_x_s,
        path_y_s,
        path_z_s,
        color="black",
        linewidth=SCIS_LINEWIDTH_PT,
        label="Actual VarQITE parameter path",
        zorder=10,
    )

    marker_lift = 0.02 * max(1e-8, z_span)
    ax.scatter(
        [traj_x[0]],
        [traj_y[0]],
        [float(energy_traj[0]) + marker_lift],
        marker="o",
        s=90,
        facecolors="green",
        linewidths=SCIS_MARKER_EDGEWIDTH_PT,
        label="Initial parameters",
        depthshade=False,
        zorder=20,
    )
    ax.scatter(
        [traj_x[-1]],
        [traj_y[-1]],
        [float(energy_traj[-1]) + marker_lift],
        marker="*",
        s=300,
        color="blue",
        edgecolors="black",
        linewidths=SCIS_MARKER_EDGEWIDTH_PT,
        label="Final parameters",
        depthshade=False,
        zorder=21,
    )

    if surface is None:
        # 3D axes rescale when artists are added, so pin the horizontal box to
        # the same padded trajectory range the original surface grid spanned.
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel(rf"$\theta_{{{param_i}}}$", fontsize=SCIS_FONT_SIZE, labelpad=6)
    ax.set_ylabel(rf"$\theta_{{{param_j}}}$", fontsize=SCIS_FONT_SIZE, labelpad=6)
    ax.set_zlabel("Energy", fontsize=SCIS_FONT_SIZE, labelpad=-2)
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


def draw_panel_c(ax: plt.Axes, plot_df: pd.DataFrame) -> None:
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
        ax.scatter(
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
    selected = plot_df[plot_df["is_selected"].map(parse_bool)]
    for label, source_name in [
        ("Selected Greedy-traj.+rerank candidate", "Greedy-traj. candidates"),
        ("Selected VarQITE+rerank candidate", "VarQITE candidates"),
    ]:
        rows = selected[selected["plot_group"] == source_name]
        if source_name in x_map and len(rows) > 0:
            ax.scatter(
                [x_map[source_name]],
                [float(pd.to_numeric(rows["energy"], errors="coerce").iloc[0])],
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
    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels(
        [compact_tick_labels.get(name, name) for name in x_map.keys()],
        rotation=0,
        ha="center",
        fontsize=SCIS_FONT_SIZE,
        linespacing=0.95,
    )
    ax.set_ylabel(r"Local decoding energy $\|r-D\delta\|^2$", fontsize=SCIS_FONT_SIZE)
    ax.set_title("(c)", fontsize=SCIS_FONT_SIZE, fontweight="normal", pad=2)
    apply_scis_axis_style(ax, grid=True, grid_axis="y")
    ax.legend(
        frameon=False,
        fontsize=SCIS_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        labelspacing=0.25,
        handletextpad=0.4,
    )


def draw_panel_d(ax: plt.Axes, plot_df: pd.DataFrame) -> plt.Axes:
    qite_plot_df = plot_df[plot_df["plot_group"] == "VarQITE candidates"].copy()

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

    ax.bar(
        rank_x,
        rank_df["energy"].to_numpy(dtype=float),
        width=0.8,
        align="center",
        alpha=0.7,
        linewidth=SCIS_LINEWIDTH_PT,
        edgecolor="black",
        label="Local decoding energy",
    )
    ax.set_xlabel("Candidate ranking after reranking", fontsize=SCIS_FONT_SIZE, labelpad=3)
    ax.set_ylabel(r"Local decoding energy $\|r-D\delta\|^2$", fontsize=SCIS_FONT_SIZE)
    tick_candidates = [v for v in [1, 5, 10, 15, 20] if v <= len(rank_df)]
    ax.set_xticks(tick_candidates)
    ax.set_title("(d)", fontsize=SCIS_FONT_SIZE, fontweight="normal", pad=2)
    apply_scis_axis_style(ax, grid=True, grid_axis="y")

    ax_r = ax.twinx()
    ax_r.set_position(ax.get_position())
    ax_r.plot(
        rank_x,
        rank_df["lwe_prior_score"].to_numpy(dtype=float),
        marker="o",
        markersize=4.0,
        linewidth=SCIS_LINEWIDTH_PT,
        label="Reranking score",
    )

    x_left = 0.5
    x_right = len(rank_df) + 0.5
    ax.set_xlim(x_left, x_right)
    ax_r.set_xlim(x_left, x_right)
    ax.margins(x=0)
    ax_r.margins(x=0)
    ax_r.set_ylabel("Reranking score", fontsize=SCIS_FONT_SIZE)
    ax_r.tick_params(
        axis="y",
        labelsize=SCIS_FONT_SIZE,
        width=SCIS_LINEWIDTH_PT,
        length=3,
        pad=1,
    )
    for spine in ax_r.spines.values():
        spine.set_linewidth(SCIS_LINEWIDTH_PT)

    selected_rows_src = rank_df[rank_df["is_selected"].map(parse_bool)]
    if len(selected_rows_src) > 0:
        selected_delta_key = delta_to_key_for_plot(selected_rows_src.iloc[0]["delta"])
        selected_rows = rank_df[rank_df["delta"].map(delta_to_key_for_plot) == selected_delta_key]
        if len(selected_rows) > 0:
            r = int(selected_rows.iloc[0]["rank"])
            ax.axvline(
                r,
                linestyle="--",
                linewidth=SCIS_LINEWIDTH_PT,
                label="Selected candidate",
            )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(
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
    return ax_r


# =========================================================
# Figure assembly
# =========================================================
def build_figure(
    data: dict[str, Any],
    surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[plt.Figure, int, int]:
    hist_df = data["hist_df"]
    plot_df = data["plot_df"]
    config = data["config"]

    traj = build_trajectory(hist_df)

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

    # Same manual alignment as the original figure: (a) sets the vertical
    # reference for (b), and (d) sets the horizontal reference for (b).
    box_a = ax_traj.get_position()
    box_d0 = ax_rank.get_position()

    right_col_x0 = box_d0.x0 - 0.004
    surface_w = box_d0.width * 0.87
    rank_w = box_d0.width * 0.91

    ax_rank.set_position([right_col_x0, box_d0.y0, rank_w, box_d0.height])
    ax_surface.set_position([right_col_x0 - 0.05, box_a.y0, surface_w, box_a.height])

    draw_panel_a(ax_traj, hist_df)
    surf, param_i, param_j = draw_panel_b(ax_surface, traj, config, surface)
    ax_surface.set_title("(b)", fontsize=SCIS_FONT_SIZE, fontweight="normal", pad=1)

    if surf is not None:
        surface_box = ax_surface.get_position()
        cax = fig.add_axes(
            [
                surface_box.x1 + 0.07,
                surface_box.y0 + 0.12 * surface_box.height,
                0.012,
                0.72 * surface_box.height,
            ]
        )
        cbar = fig.colorbar(surf, cax=cax)
        cbar.set_label(
            "Measured encoded Hamiltonian energy",
            fontsize=SCIS_FONT_SIZE,
            labelpad=4,
        )
        cbar.ax.tick_params(labelsize=SCIS_FONT_SIZE, width=SCIS_LINEWIDTH_PT, length=3)
        cbar.outline.set_linewidth(SCIS_LINEWIDTH_PT)

    draw_panel_c(ax_dist, plot_df)
    draw_panel_d(ax_rank, plot_df)

    return fig, param_i, param_j


def save_figure(fig: plt.Figure, out_base: Path, formats: list[str]) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # Do not use bbox_inches="tight": keeping the declared figsize preserves the
    # 176 mm physical width and therefore the 8 pt text size.
    for fmt in formats:
        f = fmt.lower().lstrip(".")
        path = out_base.with_suffix(f".{f}")
        if f == "png":
            fig.savefig(path, dpi=SCIS_PNG_DPI)
        elif f == "eps":
            fig.savefig(path, format="eps")
        else:
            fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


# =========================================================
# Numerical cross-check against the archived record
# =========================================================
def report_numerical_check(data: dict[str, Any], traj: dict[str, Any]) -> None:
    summary = data["summary"]
    plot_df = data["plot_df"]
    hist_df = data["hist_df"]

    base = summary.get("baseline_results", {})
    vq = summary.get("varqite_results", {})
    gt = summary.get("greedy_trajectory_rerank_results", {})

    def group_energy(name: str) -> float | None:
        rows = plot_df[plot_df["plot_group"] == name]
        if len(rows) == 0:
            return None
        return float(pd.to_numeric(rows["energy"], errors="coerce").iloc[0])

    selected = plot_df[plot_df["is_selected"].map(parse_bool)]
    sel_vq = selected[selected["plot_group"] == "VarQITE candidates"]
    sel_gt = selected[selected["plot_group"] == "Greedy-traj. candidates"]

    checks: list[tuple[str, Any, Any]] = [
        ("Babai local energy", group_energy("Babai"), base.get("initial_residual_energy")),
        ("Exhaustive local energy", group_energy("Exhaustive local enumeration"), base.get("exact_best_energy")),
        (
            "Selected VarQITE candidate energy",
            float(pd.to_numeric(sel_vq["energy"], errors="coerce").iloc[0]) if len(sel_vq) else None,
            vq.get("candidate_best_energy"),
        ),
        (
            "Selected VarQITE reranking score",
            float(pd.to_numeric(sel_vq["lwe_prior_score"], errors="coerce").iloc[0]) if len(sel_vq) else None,
            vq.get("candidate_best_lwe_prior_score"),
        ),
        (
            "Selected greedy candidate energy",
            float(pd.to_numeric(sel_gt["energy"], errors="coerce").iloc[0]) if len(sel_gt) else None,
            gt.get("candidate_best_energy"),
        ),
        (
            "VarQITE candidate pool size",
            int((plot_df["plot_group"] == "VarQITE candidates").sum()),
            vq.get("candidate_num_unique"),
        ),
    ]

    print("Numerical cross-check of plotted values against case_summary.json:")
    all_ok = True
    for label, plotted, recorded in checks:
        if plotted is None or recorded is None:
            print(f"  [skip] {label}: not available")
            continue
        ok = np.isclose(float(plotted), float(recorded), rtol=1e-9, atol=1e-9)
        all_ok = all_ok and ok
        print(f"  [{'ok' if ok else 'MISMATCH'}] {label}: plotted={plotted} archived={recorded}")

    e_before = pd.to_numeric(hist_df["E_before"], errors="coerce")
    e_after = pd.to_numeric(hist_df["E_after"], errors="coerce")
    print(
        f"  [info] trajectory: {len(hist_df)} update steps, "
        f"E_before[0]={e_before.iloc[0]:.4f}, E_after[last]={e_after.iloc[-1]:.4f}"
    )
    print(
        f"  [info] panel (b) axes: theta_{traj['param_i']} and theta_{traj['param_j']} "
        f"(largest movement range along the archived trajectory)"
    )
    print(f"  => {'all checked values agree with the archive' if all_ok else 'DISCREPANCY FOUND'}")


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Replot the archived four-panel Figure 4 from the archived case-study "
            "CSV/JSON files. Performs no simulation."
        )
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Archived case-study directory (default: {DEFAULT_DATA_DIR}).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Directory to write the regenerated figure into (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--basename",
        type=str,
        default=DEFAULT_BASENAME,
        help=f"Output file stem (default: {DEFAULT_BASENAME}).",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=["eps", "pdf", "png"],
        help="Output formats (default: eps pdf png).",
    )
    p.add_argument(
        "--surface-csv",
        type=str,
        default=None,
        help=(
            "Optional regular-grid energy surface for panel (b), with columns "
            "theta_i, theta_j, energy. The archived run did not save the surface "
            "mesh, so panel (b) shows only the archived trajectory unless this is given."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = resolve_dir(args.data_dir)
    out_dir = resolve_dir(args.out_dir)

    print("=" * 60)
    print("Replotting the archived Figure 4. No simulation is performed.")
    print(f"Archived data dir : {data_dir}")
    print(f"Output dir        : {out_dir}")
    print("=" * 60)

    data = load_archived_data(data_dir)

    surface = None
    if args.surface_csv:
        surface_path = resolve_dir(args.surface_csv)
        surface = load_surface_csv(surface_path)
        print(f"Panel (b) surface : loaded from {surface_path}")
    else:
        print(
            "Panel (b) surface : not available in the archive; drawing the archived\n"
            "                    VarQITE trajectory only. Pass --surface-csv to add a mesh."
        )

    traj = build_trajectory(data["hist_df"])
    fig, param_i, param_j = build_figure(data, surface)
    written = save_figure(fig, out_dir / args.basename, list(args.formats))

    print()
    report_numerical_check(data, traj)
    print()
    print("Written files:")
    for p in written:
        print(f"  - {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
