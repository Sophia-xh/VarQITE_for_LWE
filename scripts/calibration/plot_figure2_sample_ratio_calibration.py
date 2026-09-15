# -*- coding: utf-8 -*-
"""
plot_figure2_sample_ratio_calibration.py
========================================

Plots Figure 2 of the paper: the minimal sample ratio c = m/n at which Babai
decoding reaches success rates of 0.5, 0.7 and 0.8, together with the median and
mean reference lines that fix the working ratio c* used by the main experiment.

Input (--data-dir, default data/figure2_sample_ratio_calibration):
    threshold_by_n.csv        per-dimension c50 / c70 / c80 thresholds (plotted)
    recommended_c_star.json   median and mean reference lines
    config.json               sweep configuration, used for the annotation
    summary_by_param.csv      success rate per (n, c); auxiliary figures only

Output (--out-dir, default output/figure2_sample_ratio_calibration):
    figure2_sample_ratio_calibration.{png,pdf,eps}
        The paper figure, a single-column 90 mm x 68.4 mm panel.

Auxiliary figures and tables (--auxiliary-figures) are development diagnostics
that do not appear in the paper. Those that compare several (omega, tau)
settings additionally need the earlier coarse scan, read from --coarse-dir
(default <data-dir>/auxiliary_coarse_scan).

Usage:
    python scripts/calibration/plot_figure2_sample_ratio_calibration.py
    python scripts/calibration/plot_figure2_sample_ratio_calibration.py \
        --data-dir data/figure2_sample_ratio_calibration --out-dir figures

说明（保留原始实现说明）：
- 图的内容与原始脚本一致，只修复标题 / 参数 / legend 挤在一起的问题
- 另外提供两张“以 m 为纵轴”的辅助图：
    1) figure_threshold_lines_m
    2) figure_refinement_thresholds_m
- 这两张图直接使用 threshold_by_n.csv 里已有的 m50 / m70 / m80 列
- 以及一张 m50 的线性拟合图：
    3) figure_refinement_m50_fit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 统一论文图中的字体。
# Windows/macOS安装了Arial时优先使用Arial；
# 其他环境自动使用DejaVu Sans。
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
    }
)


# =========================================================
# SCIS 图片规范（仅在指定的两张图中通过 plt.rc_context 使用）
# =========================================================
# SCIS2026.cls 中正文宽度为 176 mm。以下两张图按最终跨栏宽度直接生成，
# 在 LaTeX 中使用 width=\textwidth 时无需再次缩放字体。
SCIS_DPI = 600
SCIS_TEXT_WIDTH_MM = 176.0
SCIS_TEXT_WIDTH_IN = SCIS_TEXT_WIDTH_MM / 25.4

# Science Bulletin 单栏图常用目标宽度。若你后续从期刊模板中确认了更精确
# 的数值，只需改这一处常量即可。
SCI_BULLETIN_SINGLE_COL_WIDTH_MM = 90.0
SCI_BULLETIN_SINGLE_COL_WIDTH_IN = SCI_BULLETIN_SINGLE_COL_WIDTH_MM / 25.4

SCIS_FONT_SIZE_PT = 8.0

# Matplotlib 的 linewidth 单位是 pt。600 dpi 下 5 px 对应：
#   5 px * 72 pt/in / 600 px/in = 0.6 pt
SCIS_LINE_WIDTH_PT = 5.0 * 72.0 / SCIS_DPI
SCIS_GRID_LINE_WIDTH_PT = 0.35
SCIS_MARKER_SIZE_PT = 3.2
SCIS_TICK_LENGTH_PT = 2.5

SCIS_RC = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif",
    ],
    "font.size": SCIS_FONT_SIZE_PT,
    "axes.labelsize": SCIS_FONT_SIZE_PT,
    "axes.titlesize": SCIS_FONT_SIZE_PT,
    "xtick.labelsize": SCIS_FONT_SIZE_PT,
    "ytick.labelsize": SCIS_FONT_SIZE_PT,
    "legend.fontsize": SCIS_FONT_SIZE_PT,
    "mathtext.fontset": "stix",
    "axes.linewidth": SCIS_LINE_WIDTH_PT,
    "lines.linewidth": SCIS_LINE_WIDTH_PT,
    "lines.markersize": SCIS_MARKER_SIZE_PT,
    "xtick.major.width": SCIS_LINE_WIDTH_PT,
    "ytick.major.width": SCIS_LINE_WIDTH_PT,
    "xtick.minor.width": SCIS_LINE_WIDTH_PT,
    "ytick.minor.width": SCIS_LINE_WIDTH_PT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
}


# =========================================================
# 0) 项目根目录
# 当前脚本位置：
#   <PROJECT_ROOT>/scripts/calibration/plot_figure2_sample_ratio_calibration.py
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# 1) 配置
# 说明：
# - calibration_data_dir 是 Figure 2 所用的精扫（refinement）数据目录
# - coarse_scan_dir 是更早的四组 (omega, tau) 粗扫描，仅辅助图需要
# =========================================================
DEFAULT_DATA_DIR = "data/figure2_sample_ratio_calibration"
DEFAULT_COARSE_SCAN_SUBDIR = "auxiliary_coarse_scan"
DEFAULT_OUT_DIR = "output/figure2_sample_ratio_calibration"
FIGURE2_BASENAME = "figure2_sample_ratio_calibration"

CONFIG = {
    "calibration_data_dir": DEFAULT_DATA_DIR,
    "coarse_scan_dir": f"{DEFAULT_DATA_DIR}/{DEFAULT_COARSE_SCAN_SUBDIR}",
    "auxiliary_figures": False,
    "save_png": True,
    "save_pdf": True,
    "save_eps": True,
    "dpi": 220,
    # 是否保留每张图的大标题（fig.suptitle / 单图 ax.title）。
    # 注意：多子图中的小标题（例如每个子图的参数设置）始终保留。
    "show_titles": False,
}


def resolve_path(path_str: str) -> Path:
    """Resolve a path given either as absolute or relative to the artifact root."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    from_root = (PROJECT_ROOT / p).resolve()
    if from_root.exists():
        return from_root
    from_cwd = p.resolve()
    if from_cwd.exists():
        return from_cwd
    return from_root


# =========================================================
# 2) 工具函数
# =========================================================
def to_jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def label_of_setting(omega: float, tau: float) -> str:
    return rf"$\omega={omega},\ \tau={tau}$"


# 给多条线使用不同 marker，避免只依赖颜色区分。
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p", "8", "d", "+", "x"]


def marker_for_index(i: int) -> str:
    return MARKER_CYCLE[i % len(MARKER_CYCLE)]


def show_main_title(cfg: dict[str, Any]) -> bool:
    """
    控制每张图的大标题是否显示。
    对于多子图：只控制 fig.suptitle，不影响每个子图的 ax.set_title。
    对于单图：控制 ax.set_title。
    """
    return bool(cfg.get("show_titles", True))


def maybe_savefig(fig, png_path: Path | None, pdf_path: Path | None, dpi: int):
    if png_path is not None:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)



def apply_scis_axes_style(ax, *, show_grid: bool = True) -> None:
    """将坐标轴线、刻度和网格统一为 SCIS 图片规范。"""
    for spine in ax.spines.values():
        spine.set_linewidth(SCIS_LINE_WIDTH_PT)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=SCIS_FONT_SIZE_PT,
        width=SCIS_LINE_WIDTH_PT,
        length=SCIS_TICK_LENGTH_PT,
        direction="out",
    )

    if show_grid:
        ax.grid(
            True,
            alpha=0.25,
            linewidth=SCIS_GRID_LINE_WIDTH_PT,
        )
    else:
        ax.grid(False)


def save_scis_figure(
    fig,
    png_path: Path | None,
    pdf_path: Path | None,
    eps_path: Path | None = None,
) -> None:
    """
    按 SCIS 要求保存指定论文图：
    - PNG：600 dpi；
    - PDF / EPS：矢量输出；
    - 不使用 bbox_inches='tight'，从而保持声明的最终物理尺寸，
      避免 LaTeX 再缩放后使 8 pt 字体发生变化。
    """
    if png_path is not None:
        fig.savefig(
            png_path,
            dpi=SCIS_DPI,
            bbox_inches=None,
            pad_inches=0,
            facecolor="white",
        )
    if pdf_path is not None:
        fig.savefig(
            pdf_path,
            bbox_inches=None,
            pad_inches=0,
            facecolor="white",
        )
    if eps_path is not None:
        fig.savefig(
            eps_path,
            format="eps",
            bbox_inches=None,
            pad_inches=0,
            facecolor="white",
        )
    plt.close(fig)


def fit_line(x: np.ndarray, y: np.ndarray):
    """
    返回一次线性拟合 y = a*x + b 的 (a, b)。
    要求有效点数 >= 2。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan

    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


# =========================================================
# 3) 读取输入
# =========================================================
def load_inputs(cfg: dict[str, Any]):
    threshold_dir = resolve_path(cfg["coarse_scan_dir"])
    refinement_dir = resolve_path(cfg["calibration_data_dir"])

    # 第一次实验（粗扫描）：只有辅助图需要，缺失时跳过
    th_summary = None
    th_threshold = None
    th_cfg = {}
    if cfg["auxiliary_figures"]:
        missing = [
            name
            for name in ("summary_by_param.csv", "threshold_by_n.csv", "config.json")
            if not (threshold_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"The auxiliary figures need the coarse scan in {threshold_dir}, "
                f"missing: {missing}. Pass --coarse-dir, or drop --auxiliary-figures "
                f"to plot only the paper figure."
            )
        th_summary = pd.read_csv(threshold_dir / "summary_by_param.csv")
        th_threshold = pd.read_csv(threshold_dir / "threshold_by_n.csv")
        with open(threshold_dir / "config.json", "r", encoding="utf-8") as f:
            th_cfg = json.load(f)

    # 第二次实验
    ref_summary = pd.read_csv(refinement_dir / "summary_by_param.csv")
    ref_threshold = pd.read_csv(refinement_dir / "threshold_by_n.csv")
    with open(refinement_dir / "recommended_c_star.json", "r", encoding="utf-8") as f:
        ref_rec = json.load(f)
    with open(refinement_dir / "config.json", "r", encoding="utf-8") as f:
        ref_cfg = json.load(f)

    return (
        threshold_dir,
        refinement_dir,
        th_summary,
        th_threshold,
        th_cfg,
        ref_summary,
        ref_threshold,
        ref_rec,
        ref_cfg,
    )


# =========================================================
# 4) 第一次实验：总览表
# =========================================================
def make_threshold_overview_table(th_threshold: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "omega", "tau", "n",
        "c50", "c70", "c80",
        "m50", "m70", "m80",
    ]
    th_threshold = ensure_numeric(th_threshold, numeric_cols)

    rows = []
    grouped = th_threshold.groupby(["omega", "tau"], dropna=False)

    for (omega, tau), g in grouped:
        row = {
            "omega": float(omega),
            "tau": float(tau),
            "n_min": int(g["n"].min()),
            "n_max": int(g["n"].max()),
        }
        for col in ["c50", "c70", "c80"]:
            vals = g[col].dropna().astype(float)
            row[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{col}_max"] = float(vals.max()) if len(vals) else np.nan
            row[f"{col}_num_valid_n"] = int(len(vals))

        for col in ["m50", "m70", "m80"]:
            vals = g[col].dropna().astype(float)
            row[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(by=["omega", "tau"]).reset_index(drop=True)


# =========================================================
# 5) 第一次实验：图 1（原图，仍以 c 为纵轴）
# =========================================================
def plot_threshold_lines(th_threshold: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]):
    df = ensure_numeric(
        th_threshold,
        ["omega", "tau", "n", "c50", "c70", "c80"]
    ).sort_values(by=["omega", "tau", "n"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)

    for ax, col, title in zip(
        axes,
        ["c50", "c70", "c80"],
        ["Threshold line for p=0.50", "Threshold line for p=0.70", "Threshold line for p=0.80"],
    ):
        for (omega, tau), g in df.groupby(["omega", "tau"], dropna=False):
            g = g.sort_values(by="n")
            ax.plot(
                g["n"], g[col],
                marker="o",
                linewidth=2,
                label=label_of_setting(float(omega), float(tau)),
            )
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel("n")
        ax.set_ylabel("minimal c = m/n")
        ax.grid(alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    if show_main_title(cfg):
        fig.suptitle(
            "Babai success-threshold calibration across four $(\\omega,\\tau)$ settings",
            fontsize=17,
            y=0.98,
        )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.76, wspace=0.25)

    maybe_savefig(
        fig,
        out_dir / "figure_threshold_lines.png" if cfg["save_png"] else None,
        out_dir / "figure_threshold_lines.pdf" if cfg["save_pdf"] else None,
        cfg["dpi"],
    )


# =========================================================
# 6) 第一次实验：新增图 1b（以 m 为纵轴）
# =========================================================
def plot_threshold_lines_m(th_threshold: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]):
    df = ensure_numeric(
        th_threshold,
        ["omega", "tau", "n", "m50", "m70", "m80"]
    ).sort_values(by=["omega", "tau", "n"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)

    setting_groups = list(df.groupby(["omega", "tau"], dropna=False))

    for ax, col, title in zip(
        axes,
        ["m50", "m70", "m80"],
        ["Threshold line for p=0.50", "Threshold line for p=0.70", "Threshold line for p=0.80"],
    ):
        for marker_idx, ((omega, tau), g) in enumerate(setting_groups):
            g = g.sort_values(by="n")
            ax.plot(
                g["n"], g[col],
                marker=marker_for_index(marker_idx),
                linewidth=2,
                label=label_of_setting(float(omega), float(tau)),
            )
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel("n")
        ax.set_ylabel("minimal m")
        ax.grid(alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    if show_main_title(cfg):
        fig.suptitle(
            "Babai success-threshold calibration across four $(\\omega,\\tau)$ settings",
            fontsize=17,
            y=0.98,
        )
        legend_y = 0.93
        top_margin = 0.76
    else:
        # 无大标题时，把 legend 上移，同时给子图标题留出空间。
        legend_y = 1.02
        top_margin = 0.80

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.subplots_adjust(top=top_margin, wspace=0.25)

    maybe_savefig(
        fig,
        out_dir / "figure_threshold_lines_m.png" if cfg["save_png"] else None,
        out_dir / "figure_threshold_lines_m.pdf" if cfg["save_pdf"] else None,
        cfg["dpi"],
    )


# =========================================================
# 7) 第一次实验：图 2（原图保持不变）
# =========================================================
def plot_success_vs_c_grid(th_summary: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]):
    """
    生成 figure_success_vs_c_grid，并按 SCIS 要求设置：
    - 最终宽度 176 mm（与 SCIS 模板的 textwidth 一致）；
    - Times New Roman 8 pt；
    - 主曲线、坐标轴和刻度线宽 0.6 pt（600 dpi 下约 5 px）；
    - PNG 以 600 dpi 输出，同时保留矢量 PDF。
    """
    df = ensure_numeric(
        th_summary,
        ["omega", "tau", "n", "m", "c_ratio", "babai_secret_success_rate"],
    ).sort_values(by=["omega", "tau", "n", "c_ratio"]).reset_index(drop=True)

    settings = sorted(
        df[["omega", "tau"]].drop_duplicates().itertuples(index=False, name=None)
    )
    n_values = sorted(df["n"].dropna().astype(int).unique())
    marker_by_n = {int(n): marker_for_index(i) for i, n in enumerate(n_values)}

    # 保留原图约 12:9.2 的长宽比，并把宽度固定为 SCIS 正文宽度 176 mm。
    success_height_in = SCIS_TEXT_WIDTH_IN * 9.2 / 12.0

    with plt.rc_context(SCIS_RC):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(SCIS_TEXT_WIDTH_IN, success_height_in),
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()

        for panel_idx, (ax, (omega, tau)) in enumerate(zip(axes, settings)):
            sub = df[
                (df["omega"] == omega)
                & (df["tau"] == tau)
            ].copy()

            for n, g in sub.groupby("n"):
                g = g.sort_values(by="c_ratio")
                ax.plot(
                    g["c_ratio"],
                    g["babai_secret_success_rate"],
                    marker=marker_by_n.get(int(n), "o"),
                    linewidth=SCIS_LINE_WIDTH_PT,
                    markersize=SCIS_MARKER_SIZE_PT,
                    markeredgewidth=SCIS_LINE_WIDTH_PT,
                    label=rf"$n = {int(n)}$",
                )

            ax.set_title(
                rf"$\omega = {float(omega):g},\ \tau = {float(tau):g}$",
                fontsize=SCIS_FONT_SIZE_PT,
                pad=2.0,
            )

            # 只在下排显示横轴标题；只在左列显示纵轴标题，减少重复文字。
            if panel_idx >= 2:
                ax.set_xlabel(r"$c = m/n$", fontsize=SCIS_FONT_SIZE_PT)
            if panel_idx % 2 == 0:
                ax.set_ylabel("Babai success rate", fontsize=SCIS_FONT_SIZE_PT)

            apply_scis_axes_style(ax, show_grid=True)

        handles, labels = axes[-1].get_legend_handles_labels()

        if show_main_title(cfg):
            fig.suptitle(
                r"Babai success rate versus $c$ across four $(\omega, \tau)$ settings",
                fontsize=SCIS_FONT_SIZE_PT,
                y=0.995,
            )
            legend_y = 0.955
            top_margin = 0.83
        else:
            legend_y = 0.985
            top_margin = 0.86

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=7,
            frameon=False,
            fontsize=SCIS_FONT_SIZE_PT,
            handlelength=1.2,
            handletextpad=0.35,
            columnspacing=0.75,
            labelspacing=0.25,
            borderaxespad=0.0,
        )

        fig.subplots_adjust(
            left=0.105,
            right=0.985,
            bottom=0.095,
            top=top_margin,
            wspace=0.18,
            hspace=0.24,
        )

        save_scis_figure(
            fig,
            out_dir / "figure_success_vs_c_grid.png" if cfg["save_png"] else None,
            out_dir / "figure_success_vs_c_grid.pdf" if cfg["save_pdf"] else None,
        )


# =========================================================
# 8) 第二次实验：表
# 现在主阈值是 p=0.5，因此距离列改为基于 c50
# =========================================================
def make_refinement_table(ref_threshold: pd.DataFrame, ref_rec: dict[str, Any]) -> pd.DataFrame:
    df = ensure_numeric(
        ref_threshold,
        ["n", "c50", "c70", "c80", "m50", "m70", "m80"]
    ).sort_values(by="n").reset_index(drop=True)

    median_c = float(ref_rec.get("recommended_c_star_median", np.nan))
    mean_c = float(ref_rec.get("recommended_c_star_mean", np.nan))
    df["distance_to_median_c50"] = (df["c50"] - median_c).abs()
    df["distance_to_mean_c50"] = (df["c50"] - mean_c).abs()
    return df


# =========================================================
# 9) 第二次实验：图 3
# refinement heatmap
# =========================================================
def plot_refinement_heatmap(ref_summary: pd.DataFrame, out_dir: Path, cfg: dict[str, Any], ref_cfg: dict[str, Any]):
    df = ensure_numeric(
        ref_summary,
        ["n", "c_ratio", "babai_secret_success_rate"]
    ).sort_values(by=["n", "c_ratio"]).reset_index(drop=True)

    pivot = df.pivot(index="n", columns="c_ratio", values="babai_secret_success_rate").sort_index()

    q = ref_cfg.get("q", "NA")
    omega = ref_cfg.get("omega", "NA")
    tau = ref_cfg.get("tau", "NA")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{float(c):.2f}" for c in pivot.columns], rotation=30)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(n)) for n in pivot.index])

    ax.set_xlabel("c = m/n")
    ax.set_ylabel("n")
    if show_main_title(cfg):
        ax.set_title(
            rf"Refinement scan: Babai success rate heatmap $(q={q},\omega={omega},\tau={tau})$",
            fontsize=16, pad=10
        )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{float(v):.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Babai success rate")
    fig.tight_layout()

    maybe_savefig(
        fig,
        out_dir / "figure_refinement_heatmap.png" if cfg["save_png"] else None,
        out_dir / "figure_refinement_heatmap.pdf" if cfg["save_pdf"] else None,
        cfg["dpi"],
    )


# =========================================================
# 10) 第二次实验：图 4（仍以 c 为纵轴）
# 现在主阈值是 p=0.5，因此辅助线改成 c50 的 median / mean
# =========================================================
def plot_refinement_thresholds(
    ref_threshold: pd.DataFrame,
    ref_rec: dict[str, Any],
    out_dir: Path,
    cfg: dict[str, Any],
    ref_cfg: dict[str, Any],
):
    """
    单独生成 figure_refinement_thresholds。

    该图直接复用 figure_refinement_thresholds_combined 的子图 (b) 的
    数据、线型、marker、legend 和统一 8 pt 字号风格，用它替换旧版
    figure_refinement_thresholds。

    这里将该图单独生成为适合 Science Bulletin 单栏排版的尺寸，
    并同时输出 PNG / PDF / EPS。
    """
    df = ensure_numeric(
        ref_threshold,
        ["n", "c50", "c70", "c80"],
    ).sort_values(by="n").reset_index(drop=True)

    median_c = float(ref_rec.get("recommended_c_star_median", np.nan))
    mean_c = float(ref_rec.get("recommended_c_star_mean", np.nan))

    q = ref_cfg.get("q", "NA")
    omega = ref_cfg.get("omega", "NA")
    tau = ref_cfg.get("tau", "NA")

    # 单独图按 Science Bulletin 单栏宽度生成。字体统一为 8 pt。
    # 当前默认单栏宽度取 90 mm；若后续模板给出更精确数值，只需修改
    # SCI_BULLETIN_SINGLE_COL_WIDTH_MM 即可。
    standalone_height_in = SCI_BULLETIN_SINGLE_COL_WIDTH_IN * 0.76

    with plt.rc_context(SCIS_RC):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(SCI_BULLETIN_SINGLE_COL_WIDTH_IN, standalone_height_in),
        )

        ax.plot(
            df["n"], df["c50"],
            marker="o",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$c_{50}$",
        )
        ax.plot(
            df["n"], df["c70"],
            marker="s",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$c_{70}$",
        )
        ax.plot(
            df["n"], df["c80"],
            marker="^",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$c_{80}$",
        )

        if not np.isnan(median_c):
            ax.axhline(
                median_c,
                linestyle="--",
                linewidth=SCIS_LINE_WIDTH_PT,
                label=rf"median$(c_{{50}}) = {median_c:.3f}$",
            )
        if not np.isnan(mean_c):
            ax.axhline(
                mean_c,
                linestyle=":",
                linewidth=SCIS_LINE_WIDTH_PT,
                label=rf"mean$(c_{{50}}) = {mean_c:.3f}$",
            )

        ax.set_xlabel(r"$n$", fontsize=SCIS_FONT_SIZE_PT)
        ax.set_ylabel(r"Minimal $c = m/n$", fontsize=SCIS_FONT_SIZE_PT)
        apply_scis_axes_style(ax, show_grid=True)
        ax.legend(
            loc="upper left",
            fontsize=SCIS_FONT_SIZE_PT,
            frameon=False,
            handlelength=1.4,
            handletextpad=0.4,
            labelspacing=0.25,
            borderaxespad=0.35,
        )

        if show_main_title(cfg):
            ax.set_title(
                rf"Refinement thresholds for $(q = {q},\ \omega = {omega},\ \tau = {tau})$",
                fontsize=SCIS_FONT_SIZE_PT,
                pad=4.0,
            )
            top_margin = 0.92
        else:
            top_margin = 0.97

        fig.subplots_adjust(
            left=0.12,
            right=0.98,
            bottom=0.14,
            top=top_margin,
        )

        save_scis_figure(
            fig,
            out_dir / f"{FIGURE2_BASENAME}.png" if cfg["save_png"] else None,
            out_dir / f"{FIGURE2_BASENAME}.pdf" if cfg["save_pdf"] else None,
            out_dir / f"{FIGURE2_BASENAME}.eps" if cfg.get("save_eps", False) else None,
        )


# =========================================================
# 11) 第二次实验：新增图 4b（以 m 为纵轴）
# 直接使用 csv 中已有的 m50 / m70 / m80
# 现在主阈值虽然是 p=0.5，但图里仍保留三条线
# =========================================================
def plot_refinement_thresholds_m(ref_threshold: pd.DataFrame, ref_rec: dict[str, Any], out_dir: Path, cfg: dict[str, Any], ref_cfg: dict[str, Any]):
    df = ensure_numeric(
        ref_threshold,
        ["n", "m50", "m70", "m80"]
    ).sort_values(by="n").reset_index(drop=True)

    q = ref_cfg.get("q", "NA")
    omega = ref_cfg.get("omega", "NA")
    tau = ref_cfg.get("tau", "NA")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["n"], df["m50"], marker="o", linewidth=2, label="p = 0.5")
    ax.plot(df["n"], df["m70"], marker="s", linewidth=2, label="p = 0.7")
    ax.plot(df["n"], df["m80"], marker="^", linewidth=2, label="p = 0.8")

    ax.set_xlabel("n")
    ax.set_ylabel("minimal m")
    if show_main_title(cfg):
        ax.set_title(
            rf"Minimal $m$ for different Babai success thresholds $(q={q},\omega={omega},\tau={tau})$",
            fontsize=16,
            pad=10,
        )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()

    maybe_savefig(
        fig,
        out_dir / "figure_refinement_thresholds_m.png" if cfg["save_png"] else None,
        out_dir / "figure_refinement_thresholds_m.pdf" if cfg["save_pdf"] else None,
        cfg["dpi"],
    )


# =========================================================
# 12) 第二次实验：新增合并图
# 把 figure_refinement_thresholds_m 和 figure_refinement_thresholds
# 合并为同一张图中的两个子图：(a) 和 (b)
# =========================================================
def plot_refinement_thresholds_combined(
    ref_threshold: pd.DataFrame,
    ref_rec: dict[str, Any],
    out_dir: Path,
    cfg: dict[str, Any],
    ref_cfg: dict[str, Any],
):
    """
    生成 figure_refinement_thresholds_combined，并按 SCIS 要求设置：
    - 最终宽度 176 mm；
    - Times New Roman 8 pt；
    - 主曲线、坐标轴和刻度线宽 0.6 pt（600 dpi 下约 5 px）；
    - PNG 以 600 dpi 输出，同时保留矢量 PDF。
    """
    df = ensure_numeric(
        ref_threshold,
        ["n", "m50", "m70", "m80", "c50", "c70", "c80"],
    ).sort_values(by="n").reset_index(drop=True)

    median_c = float(ref_rec.get("recommended_c_star_median", np.nan))
    mean_c = float(ref_rec.get("recommended_c_star_mean", np.nan))

    q = ref_cfg.get("q", "NA")
    omega = ref_cfg.get("omega", "NA")
    tau = ref_cfg.get("tau", "NA")

    # 保留原合并图约 11.5:4.8 的长宽比，并固定为 SCIS 正文宽度。
    combined_height_in = SCIS_TEXT_WIDTH_IN * 4.8 / 11.5

    with plt.rc_context(SCIS_RC):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(SCIS_TEXT_WIDTH_IN, combined_height_in),
            sharex=True,
        )
        ax_m, ax_c = axes

        # -------------------------
        # (a) minimal m
        # -------------------------
        ax_m.plot(
            df["n"], df["m50"],
            marker="o",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$p = 0.5$",
        )
        ax_m.plot(
            df["n"], df["m70"],
            marker="s",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$p = 0.7$",
        )
        ax_m.plot(
            df["n"], df["m80"],
            marker="^",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$p = 0.8$",
        )

        ax_m.set_xlabel(r"$n$", fontsize=SCIS_FONT_SIZE_PT)
        ax_m.set_ylabel(r"Minimal $m$", fontsize=SCIS_FONT_SIZE_PT)
        ax_m.set_title("(a)", fontsize=SCIS_FONT_SIZE_PT, pad=2.0)
        apply_scis_axes_style(ax_m, show_grid=True)
        ax_m.legend(
            loc="upper left",
            fontsize=SCIS_FONT_SIZE_PT,
            frameon=False,
            handlelength=1.4,
            handletextpad=0.4,
            borderaxespad=0.35,
        )

        # -------------------------
        # (b) minimal c = m/n
        # -------------------------
        ax_c.plot(
            df["n"], df["c50"],
            marker="o",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$c_{50}$",
        )
        ax_c.plot(
            df["n"], df["c70"],
            marker="s",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$c_{70}$",
        )
        ax_c.plot(
            df["n"], df["c80"],
            marker="^",
            linewidth=SCIS_LINE_WIDTH_PT,
            markersize=SCIS_MARKER_SIZE_PT,
            markeredgewidth=SCIS_LINE_WIDTH_PT,
            label=r"$c_{80}$",
        )

        if not np.isnan(median_c):
            ax_c.axhline(
                median_c,
                linestyle="--",
                linewidth=SCIS_LINE_WIDTH_PT,
                label=rf"median$(c_{{50}}) = {median_c:.3f}$",
            )
        if not np.isnan(mean_c):
            ax_c.axhline(
                mean_c,
                linestyle=":",
                linewidth=SCIS_LINE_WIDTH_PT,
                label=rf"mean$(c_{{50}}) = {mean_c:.3f}$",
            )

        ax_c.set_xlabel(r"$n$", fontsize=SCIS_FONT_SIZE_PT)
        ax_c.set_ylabel(r"Minimal $c = m/n$", fontsize=SCIS_FONT_SIZE_PT)
        ax_c.set_title("(b)", fontsize=SCIS_FONT_SIZE_PT, pad=2.0)
        apply_scis_axes_style(ax_c, show_grid=True)
        ax_c.legend(
            loc="upper left",
            fontsize=SCIS_FONT_SIZE_PT,
            frameon=False,
            handlelength=1.4,
            handletextpad=0.4,
            labelspacing=0.25,
            borderaxespad=0.35,
        )

        if show_main_title(cfg):
            fig.suptitle(
                rf"Refinement thresholds for $(q = {q},\ \omega = {omega},\ \tau = {tau})$",
                fontsize=SCIS_FONT_SIZE_PT,
                y=0.995,
            )
            top_margin = 0.88
        else:
            top_margin = 0.94

        fig.subplots_adjust(
            left=0.085,
            right=0.99,
            bottom=0.20,
            top=top_margin,
            wspace=0.29,
        )

        save_scis_figure(
            fig,
            out_dir / "figure_refinement_thresholds_combined.png" if cfg["save_png"] else None,
            out_dir / "figure_refinement_thresholds_combined.pdf" if cfg["save_pdf"] else None,
            out_dir / "figure_refinement_thresholds_combined.eps" if cfg.get("save_eps", False) else None,
        )


# =========================================================
# 13) 第二次实验：新增图 4c
# m50 原始点 + 线性拟合线
# =========================================================
def plot_refinement_m50_fit(ref_threshold: pd.DataFrame, ref_rec: dict[str, Any], out_dir: Path, cfg: dict[str, Any], ref_cfg: dict[str, Any]):
    df = ensure_numeric(ref_threshold, ["n", "m50"]).sort_values(by="n").reset_index(drop=True)

    x = df["n"].to_numpy(dtype=float)
    y = df["m50"].to_numpy(dtype=float)
    a, b = fit_line(x, y)

    q = ref_cfg.get("q", "NA")
    omega = ref_cfg.get("omega", "NA")
    tau = ref_cfg.get("tau", "NA")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    ax.scatter(x, y, s=55, marker="o", label="m50 data")

    if np.isfinite(a) and np.isfinite(b):
        xfit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        yfit = a * xfit + b
        ax.plot(
            xfit,
            yfit,
            linewidth=2.2,
            label=rf"linear fit: $m_{{50}}^{{\mathrm{{fit}}}}(n)={a:.3f}n{b:+.3f}$",
        )

        text = (
            rf"$m_{{50}}^{{\mathrm{{fit}}}}(n)={a:.3f}n{b:+.3f}$" + "\n"
            rf"$c_{{50}}^{{\mathrm{{fit}}}}(n)={a:.3f}{b:+.3f}/n$"
        )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )
    else:
        ax.text(
            0.03,
            0.97,
            "Not enough valid points for linear fit.",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )

    if "recommended_c_star_median" in ref_rec:
        c_star = float(ref_rec.get("recommended_c_star_median", np.nan))
        if np.isfinite(c_star):
            y_star = c_star * x
            ax.plot(
                x,
                y_star,
                linestyle="--",
                linewidth=1.8,
                label=rf"$m=c^*n$, $c^*={c_star:.3f}$",
            )

    ax.set_xlabel("n")
    ax.set_ylabel("m50")
    if show_main_title(cfg):
        ax.set_title(
            rf"Linear fit of $m_{{50}}$ for the refinement run $(q={q},\omega={omega},\tau={tau})$",
            fontsize=16,
            pad=10,
        )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    maybe_savefig(
        fig,
        out_dir / "figure_refinement_m50_fit.png" if cfg["save_png"] else None,
        out_dir / "figure_refinement_m50_fit.pdf" if cfg["save_pdf"] else None,
        cfg["dpi"],
    )


# =========================================================
# 14) CLI 与主函数
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Figure 2 (sample-ratio calibration) from the archived calibration data."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        metavar="DIR",
        help=(
            "Calibration data directory holding threshold_by_n.csv, "
            f"recommended_c_star.json, config.json and summary_by_param.csv (default: {DEFAULT_DATA_DIR})."
        ),
    )
    parser.add_argument(
        "--coarse-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Earlier coarse (omega, tau) scan, needed only by --auxiliary-figures "
            f"(default: <data-dir>/{DEFAULT_COARSE_SCAN_SUBDIR})."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help=f"Directory to write the figure into (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--auxiliary-figures",
        action="store_true",
        help=(
            "Also write the auxiliary development figures and tables, which are not "
            "part of the paper. Requires the coarse scan."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = dict(CONFIG)
    cfg["calibration_data_dir"] = args.data_dir
    cfg["coarse_scan_dir"] = (
        args.coarse_dir
        if args.coarse_dir is not None
        else str(Path(args.data_dir) / DEFAULT_COARSE_SCAN_SUBDIR)
    )
    cfg["auxiliary_figures"] = bool(args.auxiliary_figures)

    (
        threshold_dir,
        refinement_dir,
        th_summary,
        th_threshold,
        th_cfg,
        ref_summary,
        ref_threshold,
        ref_rec,
        ref_cfg,
    ) = load_inputs(cfg)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = [
        out_dir / f"{FIGURE2_BASENAME}.png",
        out_dir / f"{FIGURE2_BASENAME}.pdf",
        out_dir / f"{FIGURE2_BASENAME}.eps",
    ]

    # Figure 2 本身
    plot_refinement_thresholds(ref_threshold, ref_rec, out_dir, cfg, ref_cfg)

    if cfg["auxiliary_figures"]:
        aux_dir = out_dir / "auxiliary"
        aux_dir.mkdir(parents=True, exist_ok=True)

        with open(aux_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(cfg), f, ensure_ascii=False, indent=2)

        # 表 1
        threshold_overview = make_threshold_overview_table(th_threshold)
        threshold_overview.to_csv(aux_dir / "table_threshold_overview.csv", index=False, encoding="utf-8")

        # 第一次实验的图
        plot_threshold_lines(th_threshold, aux_dir, cfg)
        plot_threshold_lines_m(th_threshold, aux_dir, cfg)
        plot_success_vs_c_grid(th_summary, aux_dir, cfg)

        # 表 2
        refinement_table = make_refinement_table(ref_threshold, ref_rec)
        refinement_table.to_csv(aux_dir / "table_refinement_thresholds.csv", index=False, encoding="utf-8")

        # 第二次实验的辅助图
        plot_refinement_heatmap(ref_summary, aux_dir, cfg, ref_cfg)
        plot_refinement_thresholds_m(ref_threshold, ref_rec, aux_dir, cfg, ref_cfg)
        plot_refinement_thresholds_combined(ref_threshold, ref_rec, aux_dir, cfg, ref_cfg)
        plot_refinement_m50_fit(ref_threshold, ref_rec, aux_dir, cfg, ref_cfg)

        written.extend(
            [
                aux_dir / "config.json",
                aux_dir / "table_threshold_overview.csv",
                aux_dir / "figure_threshold_lines.png",
                aux_dir / "figure_threshold_lines_m.png",
                aux_dir / "figure_success_vs_c_grid.png",
                aux_dir / "table_refinement_thresholds.csv",
                aux_dir / "figure_refinement_heatmap.png",
                aux_dir / "figure_refinement_thresholds_m.png",
                aux_dir / "figure_refinement_thresholds_combined.png",
                aux_dir / "figure_refinement_thresholds_combined.pdf",
                aux_dir / "figure_refinement_thresholds_combined.eps",
                aux_dir / "figure_refinement_m50_fit.png",
            ]
        )

    print("\n============================================================")
    print("Done.")
    print(f"Artifact root                 : {PROJECT_ROOT}")
    print(f"Calibration data dir          : {refinement_dir}")
    if cfg["auxiliary_figures"]:
        print(f"Coarse scan dir               : {threshold_dir}")
    print(f"Output dir                    : {out_dir}")
    print("Generated files:")
    for path in written:
        print(f"  - {path}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
