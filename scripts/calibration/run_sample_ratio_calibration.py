# -*- coding: utf-8 -*-
"""
run_sample_ratio_calibration.py
===============================

Sample-ratio calibration sweep that produced the archived Figure 2 data
(data/figure2_sample_ratio_calibration/).

前置校准实验（局部精扫并行版）：
    固定主设置 (q=17, omega=1/2, tau=1/4)，
    对 c = m/n 做细粒度扫描，锁定主实验使用的固定样本比 c*。

当前版本：
1. 主设置为 tau = 0.25
2. main_threshold = 0.50
3. 使用 ProcessPoolExecutor 做并行
4. 对每个固定 n，若不同 c 映射到相同 m，则只保留第一次出现的那个 c
5. 输出 raw_per_seed.csv / summary_by_param.csv / threshold_by_n.csv / recommended_c_star.json

This sweep is purely classical but long-running (18 dimensions x 17 sample
ratios x 200 seeds). It is NOT needed to reproduce Figure 2, which is plotted
from the archived data by plot_figure2_sample_ratio_calibration.py.

A new run never overwrites the archived data; it writes a fresh timestamped
directory under the artifact root:

<ARTIFACT_ROOT>/
  output/
    sample_ratio_calibration/
      <run_name>/
        config.json
        raw_per_seed.csv
        summary_by_param.csv
        threshold_by_n.csv
        recommended_c_star.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

# =========================================================
# 0) 项目根目录
# 当前脚本位置：
#   <PROJECT_ROOT>/scripts/calibration/run_sample_ratio_calibration.py
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common.utils import centered_mod, rank_mod_prime
from algorithms.instance.lwe_instance import LWEInstance
from algorithms.lattice.hnf_basis import full_rank_basis_from_hnf
from algorithms.lattice.reduction import reduce_basis
from algorithms.lattice.babai import babai_nearest_plane
from algorithms.recovery.recover_secret import recover_secret_from_lattice_point


# =========================================================
# 1) 默认配置
# =========================================================
CONFIG = {
    # 固定主设置
    "q": 17,
    "omega": 0.5,
    "tau": 0.25,

    # 精扫 n
    "n_list": list(range(3, 21)),

    # 对 tau=1/4，c 上界需要更高一些
    "c_list": [
        1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
        1.50, 1.55, 1.60, 1.65, 1.70,
        1.75, 1.80, 1.85, 1.90, 1.95,
        2.00
    ],

    # seed 范围 [start_seed, end_seed)
    "start_seed": 0,
    "end_seed": 200,

    # Babai 基约化设置
    "reduction_method": "lll",
    "delta_lll": 0.75,

    # 阈值
    "success_threshold_list": [0.50, 0.70, 0.80],
    "main_threshold": 0.50,

    # 并行配置
    "max_workers": 5,          # None / <=0 表示自动推断
    "parallel_chunksize": 2,
    "mp_start_method": "spawn",

    "tag": "babai_ratio_refinement_tau0p25_p50_parallel",
}


# =========================================================
# 2) 数据结构
# =========================================================
@dataclass
class RawPerSeedRow:
    omega: float
    tau: float
    n: int
    m: int
    c_ratio: float
    q: int
    seed: int

    s_true_centered: list[int]
    s_true_mod_q: list[int]
    e_true: list[int]

    w_babai: list[int]
    s_hat_babai: list[int]
    babai_secret_success: bool

    elapsed_sec: float


# =========================================================
# 3) fixed-weight ternary secret / ternary noise
# =========================================================
def make_even_weight_from_relative_weight(n: int, omega: float) -> int:
    w = int(round(omega * n))
    w = max(2, w)

    if w % 2 == 1:
        if w + 1 <= n:
            w = w + 1
        else:
            w = w - 1

    if w > n:
        w = n if n % 2 == 0 else n - 1

    if w <= 0:
        w = 2 if n >= 2 else 0

    return int(w)


def sample_fixed_weight_ternary_secret(n: int, omega: float, rng: np.random.Generator) -> np.ndarray:
    w = make_even_weight_from_relative_weight(n, omega)
    if w == 0:
        raise ValueError(f"n={n}, omega={omega} 导致权重为 0，不合法。")

    idx = rng.choice(n, size=w, replace=False)
    rng.shuffle(idx)

    s = np.zeros(n, dtype=int)
    half = w // 2
    s[idx[:half]] = 1
    s[idx[half:]] = -1
    return s


def sample_symmetric_ternary_noise(m: int, tau: float, rng: np.random.Generator) -> np.ndarray:
    probs = [tau / 2.0, 1.0 - tau, tau / 2.0]
    return rng.choice([-1, 0, 1], size=m, p=probs).astype(int)


def generate_fixed_weight_lwe_instance(
    n: int,
    m: int,
    q: int,
    seed: int,
    omega: float,
    tau: float,
) -> LWEInstance:
    rng = np.random.default_rng(seed)

    while True:
        A = rng.integers(0, q, size=(m, n), dtype=int)
        if rank_mod_prime(A, q) == n:
            break

    s_centered = sample_fixed_weight_ternary_secret(n=n, omega=omega, rng=rng)
    s_mod_q = s_centered % q

    e = sample_symmetric_ternary_noise(m=m, tau=tau, rng=rng)

    c_mod_q = (A @ s_mod_q + e) % q
    c_centered = centered_mod(c_mod_q, q)
    w_true = c_centered - e

    return LWEInstance(
        A=A,
        s_centered=s_centered,
        s_mod_q=s_mod_q,
        e=e,
        c_mod_q=c_mod_q,
        c_centered=c_centered,
        w_true=w_true,
        q=q,
    )


# =========================================================
# 4) 工具函数
# =========================================================
def vec_to_list(x: np.ndarray) -> list[int]:
    return [int(v) for v in np.asarray(x).reshape(-1)]


def to_jsonable(obj):
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


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) == 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_m_from_ratio(n: int, c_ratio: float) -> int:
    m = int(math.ceil(c_ratio * n))
    return max(m, n)


def make_unique_c_m_pairs_for_n(n: int, c_list: list[float]) -> list[tuple[float, int]]:
    """
    对固定 n，把 c_list 映射成 (c_ratio, m) 对。
    若不同 c 对应到相同 m，则只保留第一次出现的那个 c。
    """
    out: list[tuple[float, int]] = []
    seen_m: set[int] = set()

    for c_ratio in c_list:
        m = compute_m_from_ratio(n=n, c_ratio=c_ratio)
        if m in seen_m:
            print(f"  skip duplicated m for n={n}: c={c_ratio:.2f} -> m={m}")
            continue
        seen_m.add(m)
        out.append((float(c_ratio), int(m)))

    return out


def resolve_max_workers(cfg: dict[str, Any]) -> int:
    requested = cfg.get("max_workers", None)
    cpu_count = os.cpu_count() or 1
    if requested is None:
        return max(1, cpu_count - 1)
    try:
        requested_int = int(requested)
    except (TypeError, ValueError):
        return max(1, cpu_count - 1)
    if requested_int <= 0:
        return max(1, cpu_count - 1)
    return max(1, min(requested_int, cpu_count))


# =========================================================
# 5) 单 seed：只跑 Babai
# =========================================================
def solve_one_seed_babai(
    omega: float,
    tau: float,
    n: int,
    c_ratio: float,
    q: int,
    seed: int,
    cfg: dict[str, Any],
) -> RawPerSeedRow:
    t0 = time.time()

    m = compute_m_from_ratio(n=n, c_ratio=c_ratio)

    inst = generate_fixed_weight_lwe_instance(
        n=n,
        m=m,
        q=q,
        seed=seed,
        omega=omega,
        tau=tau,
    )

    B, _, _ = full_rank_basis_from_hnf(inst.A, q)
    D = reduce_basis(B, method=cfg["reduction_method"], delta=cfg["delta_lll"])

    w_babai, _ = babai_nearest_plane(D, inst.c_centered)
    s_hat_babai = recover_secret_from_lattice_point(inst.A, w_babai, q)
    babai_secret_success = bool(
        np.array_equal(np.asarray(s_hat_babai) % q, np.asarray(inst.s_mod_q) % q)
    )

    elapsed_sec = float(time.time() - t0)

    return RawPerSeedRow(
        omega=float(omega),
        tau=float(tau),
        n=int(n),
        m=int(m),
        c_ratio=float(c_ratio),
        q=int(q),
        seed=int(seed),

        s_true_centered=vec_to_list(inst.s_centered),
        s_true_mod_q=vec_to_list(inst.s_mod_q),
        e_true=vec_to_list(inst.e),

        w_babai=vec_to_list(w_babai),
        s_hat_babai=vec_to_list(s_hat_babai),
        babai_secret_success=bool(babai_secret_success),

        elapsed_sec=float(elapsed_sec),
    )


def solve_one_seed_worker(args):
    omega, tau, n, c_ratio, q, seed, cfg = args
    return solve_one_seed_babai(
        omega=omega,
        tau=tau,
        n=n,
        c_ratio=c_ratio,
        q=q,
        seed=seed,
        cfg=cfg,
    )


# =========================================================
# 6) 聚合
# =========================================================
def summarize_rows(rows: list[RawPerSeedRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, int, int, float], list[RawPerSeedRow]] = {}
    for row in rows:
        key = (row.omega, row.tau, row.n, row.m, row.c_ratio)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []

    for (omega, tau, n, m, c_ratio), group in sorted(grouped.items()):
        num_seeds = len(group)
        success_rate = float(np.mean([r.babai_secret_success for r in group])) if num_seeds > 0 else 0.0
        avg_time_sec = float(np.mean([r.elapsed_sec for r in group])) if num_seeds > 0 else 0.0

        summary_rows.append(
            {
                "omega": float(omega),
                "tau": float(tau),
                "n": int(n),
                "m": int(m),
                "c_ratio": float(c_ratio),
                "q": int(group[0].q),
                "num_seeds": int(num_seeds),
                "babai_secret_success_rate": float(success_rate),
                "avg_time_sec": float(avg_time_sec),
            }
        )

    return summary_rows


def make_threshold_table(summary_rows: list[dict[str, Any]], threshold_list: list[float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, int], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (float(row["omega"]), float(row["tau"]), int(row["n"]))
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []

    for (omega, tau, n), group in sorted(grouped.items()):
        group_sorted = sorted(group, key=lambda r: (float(r["c_ratio"]), int(r["m"])))

        row_out = {
            "omega": float(omega),
            "tau": float(tau),
            "n": int(n),
        }

        for thr in threshold_list:
            chosen_c = ""
            chosen_m = ""
            chosen_success = ""

            for r in group_sorted:
                if float(r["babai_secret_success_rate"]) >= float(thr):
                    chosen_c = float(r["c_ratio"])
                    chosen_m = int(r["m"])
                    chosen_success = float(r["babai_secret_success_rate"])
                    break

            thr_name = int(round(100 * thr))
            row_out[f"c{thr_name}"] = chosen_c
            row_out[f"m{thr_name}"] = chosen_m
            row_out[f"success_at_c{thr_name}"] = chosen_success

        out_rows.append(row_out)

    return out_rows


def make_recommended_c_star(threshold_rows: list[dict[str, Any]], main_threshold: float) -> dict[str, Any]:
    """
    根据 threshold_by_n 中主阈值对应的 c 值，给出推荐 c*。
    """
    thr_name = int(round(100 * main_threshold))
    key = f"c{thr_name}"

    values = []
    n_values = []
    for row in threshold_rows:
        v = row.get(key, "")
        if v != "" and v is not None:
            values.append(float(v))
            n_values.append(int(row["n"]))

    if len(values) == 0:
        return {
            "main_threshold": float(main_threshold),
            "recommended_c_star_median": "",
            "recommended_c_star_mean": "",
            "num_valid_n": 0,
            "valid_n_list": [],
            "min_c": "",
            "max_c": "",
            "message": "没有任何 n 达到主阈值，无法给出推荐 c*。",
        }

    arr = np.asarray(values, dtype=float)

    return {
        "main_threshold": float(main_threshold),
        "recommended_c_star_median": float(np.median(arr)),
        "recommended_c_star_mean": float(np.mean(arr)),
        "num_valid_n": int(len(values)),
        "valid_n_list": n_values,
        "min_c": float(np.min(arr)),
        "max_c": float(np.max(arr)),
        "message": (
            "建议先优先参考 recommended_c_star_median。"
            "若后续主实验想保守一点，可取略大于 median 的值。"
        ),
    }


# =========================================================
# 7) 主函数
# =========================================================
def main():
    cfg = dict(CONFIG)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"q{cfg['q']}_omega{cfg['omega']}_tau{cfg['tau']}_"
        f"n{min(cfg['n_list'])}-{max(cfg['n_list'])}_"
        f"c{min(cfg['c_list'])}-{max(cfg['c_list'])}_"
        f"s{cfg['start_seed']}-{cfg['end_seed']}_"
        f"{timestamp}"
    )

    results_root_dir = PROJECT_ROOT / "output" / "sample_ratio_calibration"
    run_dir = results_root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_json_path = run_dir / "config.json"
    raw_csv_path = run_dir / "raw_per_seed.csv"
    summary_csv_path = run_dir / "summary_by_param.csv"
    threshold_csv_path = run_dir / "threshold_by_n.csv"
    recommended_json_path = run_dir / "recommended_c_star.json"

    with config_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(cfg), f, ensure_ascii=False, indent=2)

    raw_rows: list[RawPerSeedRow] = []

    omega = float(cfg["omega"])
    tau = float(cfg["tau"])
    q = int(cfg["q"])

    max_workers = resolve_max_workers(cfg)
    mp_start_method = str(cfg.get("mp_start_method", "spawn"))
    parallel_chunksize = max(1, int(cfg.get("parallel_chunksize", 1)))

    print("\n============================================================")
    print("Babai ratio refinement (parallel) starts.")
    print(f"Script path           : {Path(__file__).resolve()}")
    print(f"Project root          : {PROJECT_ROOT}")
    print(f"Results root          : {results_root_dir}")
    print(f"Run dir               : {run_dir}")
    print(f"omega                 : {omega}")
    print(f"tau                   : {tau}")
    print(f"q                     : {q}")
    print(f"main_threshold        : {cfg['main_threshold']}")
    print(f"max_workers           : {max_workers}")
    print(f"mp_start_method       : {mp_start_method}")
    print(f"parallel_chunksize    : {parallel_chunksize}")
    print("============================================================\n")

    # 为了减少频繁创建进程池的开销：
    # 这里按每个固定 n 开一个进程池，并行跑该 n 下所有 (c, seed) 任务
    for n in cfg["n_list"]:
        unique_pairs = make_unique_c_m_pairs_for_n(n=n, c_list=cfg["c_list"])

        print(f"\n===== n={n}: unique (c,m) pairs =====")
        for c_ratio, m in unique_pairs:
            print(f"  keep c={c_ratio:.2f} -> m={m}")

        task_args = []
        for c_ratio, m in unique_pairs:
            for seed in range(cfg["start_seed"], cfg["end_seed"]):
                task_args.append((omega, tau, n, c_ratio, q, seed, cfg))

        print(f"submitting {len(task_args)} tasks for n={n} ...")

        point_rows: list[RawPerSeedRow] = []
        point_t0 = time.time()

        ctx = mp.get_context(mp_start_method)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            for row in executor.map(solve_one_seed_worker, task_args, chunksize=parallel_chunksize):
                if row is None:
                    continue
                point_rows.append(row)

        point_rows.sort(key=lambda r: (int(r.c_ratio * 1000), int(r.seed)))
        raw_rows.extend(point_rows)

        elapsed_point = time.time() - point_t0
        print(f"collected {len(point_rows)} valid rows for n={n} in {elapsed_point:.2f} sec.")

    raw_rows.sort(key=lambda r: (int(r.n), int(r.m), float(r.c_ratio), int(r.seed)))
    raw_dict_rows = [asdict(r) for r in raw_rows]
    save_csv(raw_csv_path, raw_dict_rows)

    summary_rows = summarize_rows(raw_rows)
    save_csv(summary_csv_path, summary_rows)

    threshold_rows = make_threshold_table(
        summary_rows=summary_rows,
        threshold_list=cfg["success_threshold_list"],
    )
    save_csv(threshold_csv_path, threshold_rows)

    rec = make_recommended_c_star(
        threshold_rows=threshold_rows,
        main_threshold=float(cfg["main_threshold"]),
    )
    with recommended_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(rec), f, ensure_ascii=False, indent=2)

    print("\n============================================================")
    print("Done.")
    print(f"Script path           : {Path(__file__).resolve()}")
    print(f"Project root          : {PROJECT_ROOT}")
    print(f"Results root          : {results_root_dir}")
    print(f"Run dir               : {run_dir}")
    print(f"Config                : {config_json_path}")
    print(f"Raw per-seed CSV      : {raw_csv_path}")
    print(f"Summary CSV           : {summary_csv_path}")
    print(f"Threshold CSV         : {threshold_csv_path}")
    print(f"Recommended c* JSON   : {recommended_json_path}")
    print("============================================================\n")
    print("Recommended c*:")
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()