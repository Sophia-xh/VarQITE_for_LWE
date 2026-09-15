# -*- coding: utf-8 -*-
"""
plot_figure3_performance_comparison.py
=====================================

Plots Figure 3 of the paper from the archived per-seed data of

    - the main VarQITE experiment  (VarQITE candidate generation + LWE-prior reranking)
    - the greedy-trajectory baseline (greedy-trajectory candidates + the same reranking)

Figure 3 (figure3_performance_comparison) is a two-panel 158 mm x 135 mm figure:
panel (a) compares secret-recovery success rates across methods, panel (b) shows
the fair candidate-generation advantage of VarQITE over the greedy trajectory
under the identical reranking rule.

The script also re-emits the merged per-dimension table it plots, as
figure3_plot_data.csv, and writes six auxiliary development figures that do not
appear in the paper:

1. figure_main_01_success_rates_grouped_bar
   Grouped bar chart for main secret-recovery success rates.

2. figure_main_02_babai_failure_repair_grouped_bar
   Grouped bar chart for Babai-failure repair rates.

3. figure_main_03_fair_advantage_diverging_bar
   Diverging bar chart:
       - QITE candidate succeeds, greedy-trajectory candidate fails  (positive)
       - Greedy-trajectory candidate succeeds, QITE candidate fails  (negative)
   plus net advantage markers.

4. figure_main_04_runtime_scaling_logline
   Log-scale runtime scaling line plot.

5. figure_main_05_candidate_pool_boxplot
   Boxplots from raw_per_seed / merged_raw_per_seed:
       - greedy trajectory generated candidate count
       - VarQITE decoded unique candidate count
   If raw files are unavailable, falls back to summary bars.

6. figure_main_06_main_panel_mixed
   A mixed 2x2 paper-style overview panel.

Usage
-----
python scripts/performance/plot_figure3_performance_comparison.py

The defaults point at the archived data, so no arguments are required. Whole run
directories can be given instead of individual files:

python scripts/performance/plot_figure3_performance_comparison.py \
  --main_run_dir data/figure3_performance_comparison/main_experiment \
  --greedy_run_dir data/figure3_performance_comparison/greedy_baseline

or each input can be pointed at directly with --main_summary_csv,
--greedy_summary_csv, --main_config_json, --main_raw_csv and --greedy_raw_csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


# =========================================================
# 0) Project root and default input/output locations
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MAIN_EXPERIMENT_DIR = "data/figure3_performance_comparison/main_experiment"
GREEDY_BASELINE_DIR = "data/figure3_performance_comparison/greedy_baseline"
DEFAULT_OUT_DIR = "output/figure3_performance_comparison"
FIGURE3_BASENAME = "figure3_performance_comparison"
FIGURE3_PLOT_DATA_CSV = "figure3_plot_data.csv"


# =========================================================
# 1) Style
# =========================================================
def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
        }
    )


# =========================================================
# 2) CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Figure 3 (performance comparison) from the archived main-experiment "
            "and greedy-baseline data."
        )
    )

    parser.add_argument("--main_run_dir", type=str, metavar="DIR",
                        default=MAIN_EXPERIMENT_DIR,
                        help="Main VarQITE experiment run directory.")
    parser.add_argument("--greedy_run_dir", type=str, metavar="DIR",
                        default=GREEDY_BASELINE_DIR,
                        help="Greedy-trajectory baseline run directory.")

    parser.add_argument("--main_summary_csv", type=str, metavar="CSV",
                        default=f"{MAIN_EXPERIMENT_DIR}/summary_by_param.csv",
                        help="Per-dimension summary of the main experiment.")
    parser.add_argument("--greedy_summary_csv", type=str, metavar="CSV",
                        default=f"{GREEDY_BASELINE_DIR}/greedy_trajectory_fair_summary_by_param.csv",
                        help="Per-dimension fair-comparison summary of the greedy baseline.")
    parser.add_argument("--main_config_json", type=str, metavar="JSON",
                        default=f"{MAIN_EXPERIMENT_DIR}/config.json",
                        help="Main-experiment configuration, used for the figure subtitle.")

    parser.add_argument("--main_raw_csv", type=str, metavar="CSV",
                        default=f"{MAIN_EXPERIMENT_DIR}/raw_per_seed.csv",
                        help="Per-seed results of the main experiment.")
    parser.add_argument("--greedy_raw_csv", type=str, metavar="CSV",
                        default=f"{GREEDY_BASELINE_DIR}/merged_greedy_trajectory_raw_per_seed.csv",
                        help="Per-seed join of the VarQITE and greedy-baseline results.")

    parser.add_argument("--out_dir", type=str, default=None, metavar="DIR",
                        help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--x_axis", choices=["n", "m"], default="n")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no_subtitle", action="store_true")

    return parser.parse_args()


def resolve_input_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    project_candidate = (PROJECT_ROOT / p).resolve()
    cwd_candidate = p.resolve()
    if project_candidate.exists():
        return project_candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return project_candidate


def pick_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path | None, Path | None, Path]:
    main_run_dir = resolve_input_path(args.main_run_dir)
    greedy_run_dir = resolve_input_path(args.greedy_run_dir)

    # Summaries
    if args.main_summary_csv:
        main_summary = resolve_input_path(args.main_summary_csv)
    elif main_run_dir is not None:
        main_summary = main_run_dir / "summary_by_param.csv"
    else:
        raise ValueError("Please provide --main_run_dir or --main_summary_csv.")
    if main_summary is None or not main_summary.exists():
        raise FileNotFoundError(f"main-experiment summary not found: {main_summary}")

    if args.greedy_summary_csv:
        greedy_summary = resolve_input_path(args.greedy_summary_csv)
    elif greedy_run_dir is not None:
        greedy_summary = greedy_run_dir / "greedy_trajectory_fair_summary_by_param.csv"
    else:
        raise ValueError("Please provide --greedy_run_dir or --greedy_summary_csv.")
    if greedy_summary is None or not greedy_summary.exists():
        raise FileNotFoundError(f"greedy-baseline summary not found: {greedy_summary}")

    # Config
    if args.main_config_json:
        main_cfg = resolve_input_path(args.main_config_json)
    else:
        candidates = []
        if main_run_dir is not None:
            candidates.append(main_run_dir / "config.json")
        candidates.append(main_summary.parent / "config.json")
        main_cfg = pick_existing(candidates)

    # Raw files
    if args.main_raw_csv:
        main_raw = resolve_input_path(args.main_raw_csv)
    else:
        candidates = []
        if main_run_dir is not None:
            candidates.extend([
                main_run_dir / "raw_per_seed.csv",
                main_run_dir / "merged_raw_per_seed.csv",
            ])
        candidates.extend([
            main_summary.parent / "raw_per_seed.csv",
            main_summary.parent / "merged_raw_per_seed.csv",
        ])
        main_raw = pick_existing(candidates)

    if args.greedy_raw_csv:
        greedy_raw = resolve_input_path(args.greedy_raw_csv)
    else:
        candidates = []
        if greedy_run_dir is not None:
            candidates.extend([
                greedy_run_dir / "merged_greedy_trajectory_raw_per_seed.csv",
                greedy_run_dir / "greedy_trajectory_raw_per_seed.csv",
                greedy_run_dir / "merged_raw_per_seed.csv",
                greedy_run_dir / "raw_per_seed.csv",
            ])
        candidates.extend([
            greedy_summary.parent / "merged_greedy_trajectory_raw_per_seed.csv",
            greedy_summary.parent / "greedy_trajectory_raw_per_seed.csv",
            greedy_summary.parent / "merged_raw_per_seed.csv",
            greedy_summary.parent / "raw_per_seed.csv",
        ])
        greedy_raw = pick_existing(candidates)

    # Output
    out_dir = resolve_input_path(args.out_dir) if args.out_dir else (PROJECT_ROOT / DEFAULT_OUT_DIR)

    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    return main_summary, greedy_summary, main_cfg, main_raw, greedy_raw, out_dir


# =========================================================
# 3) Data loading
# =========================================================
def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def selection_label(selection: str) -> str:
    mapping = {
        "energy_lwe_prior": "energy + LWE-prior reranking",
        "lwe_prior": "LWE-prior reranking",
    }
    return mapping.get(str(selection), str(selection))


def subtitle_from_config(cfg: dict[str, Any]) -> str:
    if not cfg:
        return ""
    pieces = []
    if "q" in cfg:
        pieces.append(rf"$q={cfg['q']}$")
    if "c_star" in cfg:
        pieces.append(rf"$c^*={float(cfg['c_star']):g}$")
    if "secret_weight_total" in cfg:
        pieces.append(rf"$w={float(cfg['secret_weight_total']):g}$")
    if "noise_error_rate" in cfg:
        pieces.append(rf"$\tau={float(cfg['noise_error_rate']):g}$")
    if "candidate_topk" in cfg:
        pieces.append(rf"$k={int(cfg['candidate_topk'])}$")
    if "candidate_selection" in cfg:
        pieces.append(selection_label(str(cfg["candidate_selection"])))
    return " | ".join(pieces)


def load_and_merge_summary(main_summary: Path, greedy_summary: Path) -> pd.DataFrame:
    main_df = pd.read_csv(main_summary)
    fair = pd.read_csv(greedy_summary)

    require_columns(
        main_df,
        {
            "n",
            "m",
            "q",
            "num_seeds",
            "babai_secret_success_rate",
            "exact_local_energy_oracle_success_rate",
            "classical_local_baseline_success_rate",
            "varqite_primary_secret_success_rate",
            "varqite_candidate_lwe_prior_success_rate",
            "babai_fail_but_oracle_fix_rate",
            "babai_fail_but_classical_fix_rate",
            "babai_fail_but_qite_primary_fix_rate",
            "babai_fail_but_qite_candidate_best_fix_rate",
            "avg_candidate_num_unique",
            "avg_candidate_topk_used",
            "avg_classical_time_sec",
            "avg_qite_time_sec",
            "avg_candidate_rerank_time_sec",
        },
        "main-experiment summary",
    )

    require_columns(
        fair,
        {
            "n",
            "m",
            "q",
            "babai_secret_success_rate",
            "exact_local_energy_oracle_success_rate",
            "classical_greedy_success_rate",
            "greedy_trajectory_candidate_lwe_prior_success_rate",
            "babai_fail_but_greedy_trajectory_candidate_fix_rate",
            "babai_fail_but_qite_candidate_fix_rate",
            "qite_candidate_success_but_greedy_trajectory_candidate_fail_rate",
            "greedy_trajectory_candidate_success_but_qite_fail_rate",
            "qite_minus_greedy_trajectory_candidate_net_win_rate",
            "avg_greedy_trajectory_candidate_generated_raw",
            "avg_greedy_trajectory_candidate_topk_used",
            "avg_greedy_time_sec",
            "avg_greedy_trajectory_candidate_rerank_time_sec",
            "validation_babai_success_match_rate",
            "validation_greedy_success_match_rate",
            "validation_w_babai_match_rate",
            "validation_residual_match_rate",
        },
        "fair summary",
    )

    merged = main_df.merge(
        fair,
        on=["n", "m", "q"],
        how="inner",
        suffixes=("_exp11", "_fair"),
        validate="one_to_one",
    ).sort_values(["n", "m", "q"]).reset_index(drop=True)

    # Validation checks
    validation_cols = [
        "validation_babai_success_match_rate",
        "validation_greedy_success_match_rate",
        "validation_w_babai_match_rate",
        "validation_residual_match_rate",
    ]
    for col in validation_cols:
        if np.min(merged[col].to_numpy(dtype=float)) < 1.0 - 1e-12:
            raise ValueError(f"{col} is not identically 1.0; please inspect raw outputs.")

    # Some columns, such as num_seeds and VarQITE candidate success rate,
    # can appear in both summaries. After pandas.merge(..., suffixes=...),
    # they become *_exp11 and *_fair. Use the main-experiment version for plotting.
    num_seeds_col = "num_seeds_exp11" if "num_seeds_exp11" in merged.columns else "num_seeds"
    varqite_candidate_success_col = (
        "varqite_candidate_lwe_prior_success_rate_exp11"
        if "varqite_candidate_lwe_prior_success_rate_exp11" in merged.columns
        else "varqite_candidate_lwe_prior_success_rate"
    )

    clean = pd.DataFrame(
        {
            "n": merged["n"],
            "m": merged["m"],
            "q": merged["q"],
            "num_seeds": merged[num_seeds_col],

            "babai_success": merged["babai_secret_success_rate_exp11"],
            "exact_energy_oracle_success": merged["exact_local_energy_oracle_success_rate_exp11"],
            "greedy_success": merged["classical_local_baseline_success_rate"],
            "greedy_trajectory_prior_success": merged["greedy_trajectory_candidate_lwe_prior_success_rate"],
            "primary_varqite_success": merged["varqite_primary_secret_success_rate"],
            "varqite_candidate_prior_success": merged[varqite_candidate_success_col],
            "greedy_fix": merged["babai_fail_but_classical_fix_rate"],
            "greedy_trajectory_prior_fix": merged["babai_fail_but_greedy_trajectory_candidate_fix_rate"],
            "primary_varqite_fix": merged["babai_fail_but_qite_primary_fix_rate"],
            "varqite_candidate_prior_fix": merged["babai_fail_but_qite_candidate_best_fix_rate"],
            "qite_wins_rate": merged["qite_candidate_success_but_greedy_trajectory_candidate_fail_rate"],
            "greedy_traj_wins_rate": merged["greedy_trajectory_candidate_success_but_qite_fail_rate"],
            "qite_net_advantage": merged["qite_minus_greedy_trajectory_candidate_net_win_rate"],
            "avg_qite_candidate_unique": merged["avg_candidate_num_unique"],
            "avg_qite_candidate_topk_used": merged["avg_candidate_topk_used"],
            "avg_greedy_traj_candidate_raw": merged["avg_greedy_trajectory_candidate_generated_raw"],
            "avg_greedy_traj_candidate_topk_used": merged["avg_greedy_trajectory_candidate_topk_used"],
            "avg_greedy_runtime": merged["avg_classical_time_sec"],
            "avg_greedy_traj_rerank_runtime": merged["avg_greedy_trajectory_candidate_rerank_time_sec"],
            "avg_qite_runtime": merged["avg_qite_time_sec"],
            "avg_qite_rerank_runtime": merged["avg_candidate_rerank_time_sec"],
        }
    )

    return clean


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_raw_for_pool_plot(main_raw: Path | None, greedy_raw: Path | None) -> pd.DataFrame | None:
    """
    Build a compact raw dataframe for the candidate-pool richness plot.

    We prefer the merged greedy-baseline raw file because it often already
    contains the main-experiment columns plus the baseline columns. If it is not
    available, we try to merge the two raw files on (n,m,q,seed).
    """
    main_df = pd.read_csv(main_raw) if (main_raw is not None and main_raw.exists()) else None
    fair_df = pd.read_csv(greedy_raw) if (greedy_raw is not None and greedy_raw.exists()) else None

    raw = None
    if fair_df is not None:
        # Often the fair merged raw file already contains everything.
        if {"n", "m", "q", "seed"}.issubset(fair_df.columns):
            raw = fair_df.copy()

    if raw is None and main_df is not None and fair_df is not None:
        key = ["n", "m", "q", "seed"]
        if set(key).issubset(main_df.columns) and set(key).issubset(fair_df.columns):
            raw = main_df.merge(fair_df, on=key, how="inner", suffixes=("_exp11", "_fair"))

    if raw is None and main_df is not None:
        raw = main_df.copy()

    if raw is None:
        return None

    key_cols = ["n", "m", "q"]
    if "seed" in raw.columns:
        key_cols.append("seed")
    if not {"n", "m"}.issubset(raw.columns):
        return None

    qite_col = first_existing_column(
        raw,
        [
            "candidate_num_unique",
            "avg_candidate_num_unique",
            "candidate_num_unique_exp11",
        ],
    )
    greedy_col = first_existing_column(
        raw,
        [
            "classical_candidate_generated_raw",
            "greedy_trajectory_candidate_generated_raw",
            "classical_single_candidate_generated_raw",
            "candidate_generated_raw",
        ],
    )

    if qite_col is None or greedy_col is None:
        return None

    compact = raw[["n", "m"]].copy()
    if "seed" in raw.columns:
        compact["seed"] = raw["seed"]
    compact["qite_unique_candidates"] = pd.to_numeric(raw[qite_col], errors="coerce")
    compact["greedy_trajectory_candidates"] = pd.to_numeric(raw[greedy_col], errors="coerce")
    compact = compact.dropna()
    return compact


# =========================================================
# 4) Helpers
# =========================================================
def x_axis_data(df: pd.DataFrame, x_col: str) -> tuple[np.ndarray, str]:
    if x_col == "n":
        return df["n"].to_numpy(dtype=float), r"LWE dimension $n$"
    return df["m"].to_numpy(dtype=float), r"Sample size $m$"


def add_title_and_subtitle(fig: plt.Figure, title: str, subtitle: str, no_subtitle: bool) -> None:
    """
    Do not add figure-level title or configuration subtitle.

    The paper figures now rely on captions in the manuscript.
    Subplot titles, such as (a)--(d) in the mixed panel, are kept inside
    the corresponding plotting functions.
    """
    return


def finish_layout(fig: plt.Figure) -> None:
    # No figure-level title/subtitle is used, so the plot area can use the full canvas.
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.98])


def save_figure(fig: plt.Figure, base_path: Path, formats: list[str]) -> None:
    for fmt in formats:
        path = base_path.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")


def configure_log_axis(ax: plt.Axes, x: np.ndarray, x_label: str) -> None:
    ax.set_xlabel(x_label)
    ax.set_xticks(x)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.22)


# High-contrast colors with clearly different lightness levels.
# These colors are chosen so that categories remain distinguishable after
# conversion to grayscale, without using hatch patterns.
BAR_COLORS = [
    "#000000",  # black, very dark
    "#5E3C99",  # purple, dark-mid
    "#1B9E77",  # green, mid
    "#E66101",  # orange, light-mid
    "#F0E442",  # yellow, very light
    "#A6CEE3",  # pale blue, light
]


def style_for_series(i: int) -> str:
    return BAR_COLORS[i % len(BAR_COLORS)]


def draw_styled_bar(ax: plt.Axes, x, height, *, width: float, label: str, series_idx: int, alpha: float = 0.92):
    color = style_for_series(series_idx)
    return ax.bar(
        x,
        height,
        width=width,
        label=label,
        color=color,
        edgecolor="black",
        linewidth=0.45,
        alpha=alpha,
    )


def style_boxplot(bp: dict[str, Any], *, facecolor: str) -> None:
    for patch in bp.get("boxes", []):
        patch.set_facecolor(facecolor)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_alpha(0.88)
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp.get(key, []):
            artist.set_color("black")
            artist.set_linewidth(1.0 if key != "medians" else 1.5)

def plot_candidate_pool_axis(
    ax: plt.Axes,
    raw_df: pd.DataFrame | None,
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    legend_fontsize: float = 7.6,
) -> None:
    """Draw the candidate-pool richness plot on a supplied axis."""
    x_vals = summary_df[args.x_axis].to_numpy(dtype=int)
    x_label = r"LWE dimension $n$" if args.x_axis == "n" else r"Sample size $m$"

    if raw_df is not None and args.x_axis in raw_df.columns:
        positions_greedy = np.arange(len(x_vals)) * 2.2
        positions_qite = positions_greedy + 0.8

        greedy_data = []
        qite_data = []
        for xv in x_vals:
            subset = raw_df.loc[raw_df[args.x_axis] == xv]
            greedy_data.append(subset["greedy_trajectory_candidates"].to_numpy(dtype=float))
            qite_data.append(subset["qite_unique_candidates"].to_numpy(dtype=float))

        b1 = ax.boxplot(
            greedy_data,
            positions=positions_greedy,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"linewidth": 1.4},
        )
        b2 = ax.boxplot(
            qite_data,
            positions=positions_qite,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"linewidth": 1.4},
        )

        color_greedy = style_for_series(1)
        color_qite = style_for_series(4)
        style_boxplot(b1, facecolor=color_greedy)
        style_boxplot(b2, facecolor=color_qite)

        tick_positions = (positions_greedy + positions_qite) / 2.0
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(v)) for v in x_vals])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Candidate count")
        ax.set_ylim(bottom=0.0)

        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=color_greedy, edgecolor="black", alpha=0.88, label="Greedy trajectory generated candidates"),
            Patch(facecolor=color_qite, edgecolor="black", alpha=0.88, label="VarQITE decoded unique candidates"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=legend_fontsize)
        return

    idx = np.arange(len(x_vals))
    width = 0.24
    fallback_series = [
        ("Greedy trajectory generated candidates", summary_df["avg_greedy_traj_candidate_raw"].to_numpy()),
        ("VarQITE decoded unique candidates", summary_df["avg_qite_candidate_unique"].to_numpy()),
        ("VarQITE retained top-$k$ candidates", summary_df["avg_qite_candidate_topk_used"].to_numpy()),
    ]
    offsets = [-width, 0.0, width]
    for i, (offset, (label, y)) in enumerate(zip(offsets, fallback_series)):
        draw_styled_bar(ax, idx + offset, y, width=width, label=label, series_idx=i + 1, alpha=0.9)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(v)) for v in x_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Average candidate count")
    ax.legend(loc="upper left", frameon=True, fontsize=legend_fontsize)


def plot_fair_advantage_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    legend_fontsize: float = 8.8,
    title: str | None = None,
) -> None:
    """
    Draw the fair candidate-generation advantage panel.

    This helper is used by both:
      - figure_main_03_fair_advantage_diverging_bar
      - subplot (c) in figure_main_06_main_panel_mixed
    so the two panels have the same data mapping, colors, bar width,
    axis limits, and legend semantics.
    """
    x_vals = df[args.x_axis].to_numpy(dtype=float)
    x_label = r"LWE dimension $n$" if args.x_axis == "n" else r"Sample size $m$"
    x_idx = np.arange(len(x_vals))
    x_ticks = [str(int(v)) for v in x_vals]

    qite_wins = df["qite_wins_rate"].to_numpy(dtype=float)
    greedy_wins = df["greedy_traj_wins_rate"].to_numpy(dtype=float)
    net = df["qite_net_advantage"].to_numpy(dtype=float)

    qite_color = style_for_series(3)
    greedy_color = style_for_series(1)
    width = 0.42

    ax.bar(
        x_idx,
        qite_wins,
        width=width,
        label=r"Our scheme wins $P_{\mathrm{VQ}>\mathrm{G}}$",
        color=qite_color,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
    )
    ax.bar(
        x_idx,
        -greedy_wins,
        width=width,
        label=r"Greedy-traj. wins $P_{\mathrm{G}>\mathrm{VQ}}$",
        color=greedy_color,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
    )
    ax.plot(
        x_idx,
        net,
        marker="o",
        linestyle="--",
        linewidth=1.8,
        color="black",
        label=r"Net advantage $\Delta_{\mathrm{net}}$",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)

    ymax = max(0.32, float(max(qite_wins.max(), greedy_wins.max(), net.max())) * 1.18)
    ymin = -max(0.05, float(greedy_wins.max()) * 1.8)
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Instance rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=legend_fontsize, frameon=True)


# =========================================================
# 5) Figure 1: grouped bar success rates
# =========================================================
def figure_success_rates_grouped_bar(df: pd.DataFrame, subtitle: str, args: argparse.Namespace, out_dir: Path) -> None:
    x_vals = df[args.x_axis].to_numpy(dtype=float)
    x_label = r"LWE dimension $n$" if args.x_axis == "n" else r"Sample size $m$"

    series = [
        ("Babai", df["babai_success"].to_numpy()),
        ("Exhaustive local enumeration", df["exact_energy_oracle_success"].to_numpy()),
        ("Greedy-traj.+rerank", df["greedy_trajectory_prior_success"].to_numpy()),
        ("VarQITE only", df["primary_varqite_success"].to_numpy()),
        ("Our scheme", df["varqite_candidate_prior_success"].to_numpy()),
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    idx = np.arange(len(x_vals))
    width = 0.15
    offsets = np.linspace(-2, 2, len(series)) * width

    for series_idx, (offset, (label, y)) in enumerate(zip(offsets, series)):
        draw_styled_bar(ax, idx + offset, y, width=width, label=label, series_idx=series_idx, alpha=0.92)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(v)) for v in x_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Secret-recovery success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.legend(loc="lower left", ncol=2, frameon=True)

    add_title_and_subtitle(fig, "Secret-recovery success rates", subtitle, args.no_subtitle)
    finish_layout(fig)
    save_figure(fig, out_dir / "figure_main_01_success_rates_grouped_bar", args.formats)
    if args.show:
        plt.show()
    plt.close(fig)


# =========================================================
# 6) Figure 2: grouped bar repair rates
# =========================================================
def figure_repair_rates_grouped_bar(df: pd.DataFrame, subtitle: str, args: argparse.Namespace, out_dir: Path) -> None:
    x_vals = df[args.x_axis].to_numpy(dtype=float)
    x_label = r"LWE dimension $n$" if args.x_axis == "n" else r"Sample size $m$"

    series = [
        ("Greedy local descent", df["greedy_fix"].to_numpy()),
        ("Greedy trajectory + LWE-prior", df["greedy_trajectory_prior_fix"].to_numpy()),
        ("Primary VarQITE", df["primary_varqite_fix"].to_numpy()),
        ("VarQITE candidates + LWE-prior", df["varqite_candidate_prior_fix"].to_numpy()),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    idx = np.arange(len(x_vals))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(series)) * width

    for series_idx, (offset, (label, y)) in enumerate(zip(offsets, series)):
        draw_styled_bar(ax, idx + offset, y, width=width, label=label, series_idx=series_idx, alpha=0.92)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(v)) for v in x_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Babai-failure repair rate\n(rate over all instances)")
    upper = max(0.32, float(df["varqite_candidate_prior_fix"].max()) * 1.22)
    ax.set_ylim(0.0, upper)
    ax.set_yticks(np.arange(0.0, upper + 1e-12, 0.05))
    ax.legend(loc="upper right", ncol=1, frameon=True)

    add_title_and_subtitle(fig, "Repairing Babai failures", subtitle, args.no_subtitle)
    finish_layout(fig)
    save_figure(fig, out_dir / "figure_main_02_babai_failure_repair_grouped_bar", args.formats)
    if args.show:
        plt.show()
    plt.close(fig)


# =========================================================
# 7) Figure 3: diverging bar fair advantage
# =========================================================
def figure_fair_advantage_diverging_bar(df: pd.DataFrame, subtitle: str, args: argparse.Namespace, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    plot_fair_advantage_axis(ax, df, args, legend_fontsize=8.8, title=None)

    add_title_and_subtitle(fig, "Fair candidate-generation comparison", subtitle, args.no_subtitle)
    finish_layout(fig)
    save_figure(fig, out_dir / "figure_main_03_fair_advantage_diverging_bar", args.formats)
    if args.show:
        plt.show()
    plt.close(fig)


# =========================================================
# 8) Figure 4: runtime scaling (log line)
# =========================================================
def figure_runtime_logline(df: pd.DataFrame, subtitle: str, args: argparse.Namespace, out_dir: Path) -> None:
    x, x_label = x_axis_data(df, args.x_axis)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(x, df["avg_greedy_runtime"].to_numpy(), marker="^", label="Greedy local descent")
    ax.plot(x, df["avg_greedy_traj_rerank_runtime"].to_numpy(), marker="s", label="Greedy-trajectory LWE-prior reranking")
    ax.plot(x, df["avg_qite_runtime"].to_numpy(), marker="D", label="VarQITE candidate generation")
    ax.plot(x, df["avg_qite_rerank_runtime"].to_numpy(), marker="o", label="VarQITE candidate LWE-prior reranking")

    ax.set_ylabel("Average runtime per instance (s)")
    configure_log_axis(ax, x, x_label)
    ax.legend(loc="upper left", frameon=True)

    add_title_and_subtitle(fig, "Runtime scaling", subtitle, args.no_subtitle)
    finish_layout(fig)
    save_figure(fig, out_dir / "figure_main_04_runtime_scaling_logline", args.formats)
    if args.show:
        plt.show()
    plt.close(fig)


# =========================================================
# 9) Figure 5: raw-based boxplot for candidate pool richness
# =========================================================
def figure_candidate_pool_richness(raw_df: pd.DataFrame | None, summary_df: pd.DataFrame, subtitle: str, args: argparse.Namespace, out_dir: Path) -> None:
    x_vals = summary_df[args.x_axis].to_numpy(dtype=int)
    x_label = r"LWE dimension $n$" if args.x_axis == "n" else r"Sample size $m$"

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    plot_candidate_pool_axis(ax, raw_df, summary_df, args, legend_fontsize=8.8)
    ax.set_xlabel(x_label)

    add_title_and_subtitle(fig, "Candidate-pool richness", subtitle, args.no_subtitle)
    finish_layout(fig)
    save_figure(fig, out_dir / "figure_main_05_candidate_pool_boxplot", args.formats)
    if args.show:
        plt.show()
    plt.close(fig)


# =========================================================
# 10) Figure 6: mixed overview panel
# =========================================================
def figure_main_panel_mixed(summary_df: pd.DataFrame, raw_df: pd.DataFrame | None, subtitle: str, args: argparse.Namespace, out_dir: Path) -> None:
    x = summary_df[args.x_axis].to_numpy(dtype=float)
    x_idx = np.arange(len(x))
    x_ticks = [str(int(v)) for v in x]
    x_label = r"LWE dimension $n$" if args.x_axis == "n" else r"Sample size $m$"

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # (a) grouped bars: success
    width = 0.14
    success_series = [
        ("Babai", summary_df["babai_success"].to_numpy()),
        ("Exhaustive local enumeration", summary_df["exact_energy_oracle_success"].to_numpy()),
        ("Greedy-traj.+rerank", summary_df["greedy_trajectory_prior_success"].to_numpy()),
        ("VarQITE only", summary_df["primary_varqite_success"].to_numpy()),
        ("Our scheme", summary_df["varqite_candidate_prior_success"].to_numpy()),
    ]
    offsets = np.linspace(-2, 2, len(success_series)) * width
    for series_idx, (offset, (label, y)) in enumerate(zip(offsets, success_series)):
        draw_styled_bar(ax1, x_idx + offset, y, width=width, label=label, series_idx=series_idx, alpha=0.9)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_ticks)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Success rate")
    ax1.set_title("(a) Secret-recovery success")
    ax1.legend(loc="lower left", fontsize=7.4, ncol=2, frameon=True)

    # (b) grouped bars: repair
    repair_series = [
        ("Greedy", summary_df["greedy_fix"].to_numpy()),
        ("Greedy traj. + prior", summary_df["greedy_trajectory_prior_fix"].to_numpy()),
        ("Primary QITE", summary_df["primary_varqite_fix"].to_numpy()),
        ("QITE cand. + prior", summary_df["varqite_candidate_prior_fix"].to_numpy()),
    ]
    width2 = 0.18
    offsets2 = np.linspace(-1.5, 1.5, len(repair_series)) * width2
    for series_idx, (offset, (label, y)) in enumerate(zip(offsets2, repair_series)):
        draw_styled_bar(ax2, x_idx + offset, y, width=width2, label=label, series_idx=series_idx, alpha=0.9)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_ticks)
    ax2.set_ylim(0.0, max(0.32, float(summary_df["varqite_candidate_prior_fix"].max()) * 1.22))
    ax2.set_ylabel("Repair rate")
    ax2.set_title("(b) Repairing Babai failures")
    ax2.legend(loc="upper right", fontsize=7.6, frameon=True)

    # (c) diverging bars: fair advantage
    plot_fair_advantage_axis(
        ax3,
        summary_df,
        args,
        legend_fontsize=7.6,
        title="(c) Fair candidate-generation advantage",
    )

    # (d) candidate-pool richness, replacing the runtime panel
    plot_candidate_pool_axis(ax4, raw_df, summary_df, args, legend_fontsize=7.4)
    ax4.set_title("(d) Candidate-pool richness")

    for ax in axes.ravel():
        ax.set_xlabel(x_label)

    add_title_and_subtitle(fig, "Main experiment overview", subtitle, args.no_subtitle)
    finish_layout(fig)
    save_figure(fig, out_dir / "figure_main_06_main_panel_mixed", args.formats)
    if args.show:
        plt.show()
    plt.close(fig)

def move_xlabel_to_right(ax: plt.Axes, y: float = -0.11) -> None:
    ax.xaxis.set_label_coords(1.0, y)
    ax.xaxis.label.set_horizontalalignment("right")


# =========================================================
# 10.5) Figure 3: vertical two-panel figure for the paper
# =========================================================
def figure_3_success_and_fair_vertical(
    summary_df: pd.DataFrame,
    subtitle: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    """
    Generate the Figure 3 canvas with two vertically stacked panels:
      (a) Secret-recovery success
      (b) Fair candidate-generation advantage

    Elsevier-compatible figure settings used only for this figure:
      - target artwork size: 158 mm x 135 mm;
      - all symbols and text: 8 pt Times/Times New Roman;
      - principal line width: 0.6 pt;
      - preferred submission output: EPS vector artwork with embedded Type 42 fonts;
      - PDF vector output is retained for LaTeX/workflow convenience;
      - PNG is retained only as a 600 dpi preview;
      - TIFF, when explicitly requested, is written at 1000 dpi for line artwork.

    Other figures generated by this script retain their original settings.
    """
    # ---------------------------------------------------------
    # Elsevier artwork dimensions and typography
    # 0.6 pt * 600 dpi / 72 pt per inch = 5 px
    # ---------------------------------------------------------
    scis_width_in = 158.0 / 25.4
    scis_height_in = 135.0 / 25.4
    scis_fontsize = 8.0
    scis_linewidth = 0.6
    scis_grid_linewidth = 0.4
    scis_marker_size = 3.4

    scis_rc = {
        # Elsevier artwork guide permits Times.  Do not list non-approved
        # fallback families here; make sure Times New Roman or Times is
        # installed on the machine that generates the final artwork.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "font.size": scis_fontsize,
        "axes.titlesize": scis_fontsize,
        "axes.labelsize": scis_fontsize,
        "legend.fontsize": scis_fontsize,
        "xtick.labelsize": scis_fontsize,
        "ytick.labelsize": scis_fontsize,
        # Keep math text visually consistent with Times rather than STIX.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,
        "axes.linewidth": scis_linewidth,
        "lines.linewidth": scis_linewidth,
        "lines.markersize": scis_marker_size,
        "grid.linewidth": scis_grid_linewidth,
        "grid.alpha": 0.22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }

    x = summary_df[args.x_axis].to_numpy(dtype=float)
    x_idx = np.arange(len(x))
    x_ticks = [str(int(v)) for v in x]
    x_label = (
        r"LWE dimension $n$"
        if args.x_axis == "n"
        else r"Sample size $m$"
    )

    with plt.rc_context(scis_rc):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(scis_width_in, scis_height_in),
            gridspec_kw={"height_ratios": [1.05, 1.0]},
        )
        ax1, ax2 = axes

        # -----------------------------------------------------
        # (a) Secret-recovery success
        # -----------------------------------------------------
        width = 0.13
        success_series = [
            ("Babai", summary_df["babai_success"].to_numpy(dtype=float)),
            (
                "Exhaustive local enumeration",
                summary_df["exact_energy_oracle_success"].to_numpy(dtype=float),
            ),
            (
                "Greedy-traj.+rerank",
                summary_df["greedy_trajectory_prior_success"].to_numpy(dtype=float),
            ),
            (
                "VarQITE only",
                summary_df["primary_varqite_success"].to_numpy(dtype=float),
            ),
            (
                "Our scheme",
                summary_df["varqite_candidate_prior_success"].to_numpy(dtype=float),
            ),
        ]
        offsets = np.linspace(-2, 2, len(success_series)) * width

        for series_idx, (offset, (label, y)) in enumerate(
            zip(offsets, success_series)
        ):
            ax1.bar(
                x_idx + offset,
                y,
                width=width,
                label=label,
                color=style_for_series(series_idx),
                edgecolor="black",
                linewidth=scis_linewidth,
                alpha=0.92,
            )

        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(x_ticks)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(
            r"Secret-recovery success rate $P_{\mathrm{succ}}(\mathcal{M})$"
        )
        ax1.set_ylim(0.0, 1.05)
        ax1.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax1.set_title("(a) Secret-recovery success", pad=4)
        ax1.grid(True, axis="y")
        ax1.set_axisbelow(True)
        ax1.legend(
            loc="lower left",
            ncol=2,
            frameon=True,
            fontsize=scis_fontsize,
            borderpad=0.35,
            labelspacing=0.25,
            handlelength=1.3,
            handletextpad=0.45,
            columnspacing=0.9,
        )

        # -----------------------------------------------------
        # (b) Fair candidate-generation advantage
        # Drawn here instead of calling the general helper so
        # that every line follows the 0.6 pt specification.
        # -----------------------------------------------------
        qite_wins = summary_df["qite_wins_rate"].to_numpy(dtype=float)
        greedy_wins = summary_df["greedy_traj_wins_rate"].to_numpy(dtype=float)
        net = summary_df["qite_net_advantage"].to_numpy(dtype=float)

        qite_color = style_for_series(3)
        greedy_color = style_for_series(1)
        fair_width = 0.36

        ax2.bar(
            x_idx,
            qite_wins,
            width=fair_width,
            label=r"Our scheme wins $P_{\mathrm{VQ}>\mathrm{G}}$",
            color=qite_color,
            edgecolor="black",
            linewidth=scis_linewidth,
            alpha=0.90,
        )
        ax2.bar(
            x_idx,
            -greedy_wins,
            width=fair_width,
            label=r"Greedy-traj. wins $P_{\mathrm{G}>\mathrm{VQ}}$",
            color=greedy_color,
            edgecolor="black",
            linewidth=scis_linewidth,
            alpha=0.90,
        )
        ax2.plot(
            x_idx,
            net,
            marker="o",
            linestyle="--",
            linewidth=scis_linewidth,
            markersize=scis_marker_size,
            markeredgewidth=scis_linewidth,
            color="black",
            label=r"Net advantage $\Delta_{\mathrm{net}}$",
        )
        ax2.axhline(0.0, color="black", linewidth=scis_linewidth)

        ymax = max(
            0.32,
            float(max(qite_wins.max(), greedy_wins.max(), net.max())) * 1.18,
        )
        ymin = -max(0.05, float(greedy_wins.max()) * 1.8)
        ax2.set_ylim(ymin, ymax)

        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(x_ticks)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("Instance rate")
        ax2.set_title("(b) Fair candidate-generation advantage", pad=4)
        ax2.grid(True, axis="y")
        ax2.set_axisbelow(True)
        ax2.legend(
            loc="upper right",
            frameon=True,
            fontsize=scis_fontsize,
            borderpad=0.35,
            labelspacing=0.25,
            handlelength=1.8,
            handletextpad=0.45,
        )

        # Apply 0.6 pt to axes, ticks, and legend frames.
        for ax in (ax1, ax2):
            for spine in ax.spines.values():
                spine.set_linewidth(scis_linewidth)
            ax.tick_params(
                axis="both",
                which="both",
                width=scis_linewidth,
                length=3.0,
                labelsize=scis_fontsize,
            )
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_linewidth(scis_linewidth)

        # Fixed margins preserve the intended physical canvas size and prevent
        # bbox-based cropping from changing the effective artwork dimensions.
        # Slightly larger left/bottom margins compensate for keeping text at 8 pt
        # while reducing the overall figure dimensions.
        fig.subplots_adjust(
            left=0.125,
            right=0.985,
            bottom=0.085,
            top=0.970,
            hspace=0.27,
        )

        base_path = out_dir / FIGURE3_BASENAME

        # Always create an EPS submission copy for Figure 3, while keeping
        # the user-requested formats (PNG/PDF by default) for convenience.
        figure3_formats = list(dict.fromkeys(["eps", *args.formats]))
        for fmt in figure3_formats:
            fmt_lower = str(fmt).lower().lstrip(".")
            save_path = base_path.with_suffix(f".{fmt_lower}")

            if fmt_lower == "eps":
                # Preferred Elsevier format for vector line artwork.
                # ps.fonttype=42 above embeds TrueType outlines in the EPS.
                fig.savefig(
                    save_path,
                    format="eps",
                    facecolor="white",
                    transparent=False,
                )
            elif fmt_lower == "pdf":
                # Vector copy for manuscript/LaTeX workflow.
                # pdf.fonttype=42 above embeds TrueType fonts.
                fig.savefig(
                    save_path,
                    format="pdf",
                    facecolor="white",
                    transparent=False,
                )
            elif fmt_lower in {"tif", "tiff"}:
                # If TIFF is requested for this line-art figure, use 1000 dpi
                # according to the Elsevier artwork guidance for bitmap line art.
                fig.savefig(
                    save_path,
                    dpi=1000,
                    facecolor="white",
                    transparent=False,
                )
            elif fmt_lower == "png":
                # High-resolution preview; EPS remains the preferred submission file.
                fig.savefig(
                    save_path,
                    dpi=600,
                    facecolor="white",
                    transparent=False,
                )
            else:
                fig.savefig(
                    save_path,
                    dpi=600,
                    facecolor="white",
                    transparent=False,
                )
            print(f"Saved: {save_path}")

        if args.show:
            plt.show()
        plt.close(fig)


# =========================================================
# 11) Main
# =========================================================
def main() -> None:
    args = parse_args()
    setup_matplotlib()

    main_summary, greedy_summary, main_cfg_path, main_raw, greedy_raw, out_dir = resolve_paths(args)
    cfg = load_config(main_cfg_path)
    subtitle = subtitle_from_config(cfg)

    summary_df = load_and_merge_summary(main_summary, greedy_summary)
    raw_df = load_raw_for_pool_plot(main_raw, greedy_raw)

    summary_df.to_csv(out_dir / FIGURE3_PLOT_DATA_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_dir / FIGURE3_PLOT_DATA_CSV}")
    if raw_df is not None:
        raw_df.to_csv(out_dir / "candidate_pool_raw_data_used.csv", index=False, encoding="utf-8-sig")
        print(f"Saved: {out_dir / 'candidate_pool_raw_data_used.csv'}")

    figure_success_rates_grouped_bar(summary_df, subtitle, args, out_dir)
    figure_repair_rates_grouped_bar(summary_df, subtitle, args, out_dir)
    figure_fair_advantage_diverging_bar(summary_df, subtitle, args, out_dir)
    figure_runtime_logline(summary_df, subtitle, args, out_dir)
    figure_candidate_pool_richness(raw_df, summary_df, subtitle, args, out_dir)
    figure_main_panel_mixed(summary_df, raw_df, subtitle, args, out_dir)
    figure_3_success_and_fair_vertical(summary_df, subtitle, args, out_dir)

    print("\nDone.")
    print(f"main summary   : {main_summary}")
    print(f"greedy summary : {greedy_summary}")
    print(f"main raw       : {main_raw}")
    print(f"greedy raw     : {greedy_raw}")
    print(f"output dir     : {out_dir}")


if __name__ == "__main__":
    main()
