# -*- coding: utf-8 -*-
"""
run_main_varqite_experiment.py
==============================

Main performance experiment that produced the archived Figure 3 data
(data/figure3_performance_comparison/main_experiment/).

主实验（scaling 并行版，VarQITE candidate generation + LWE-prior reranking）：

在 calibration 得到的固定工作区间下，按 n 扫描，统一比较：
    - Babai
    - exact local energy oracle
    - classical local baseline (greedy energy descent)
    - local VarQITE with candidate generation + LWE-prior reranking

同时保留 primary VarQITE 与 candidate-any diagnostic upper bound，便于分析：
    - primary VarQITE: 只使用 qite_res.delta_qite
    - candidate-best: 从 VarQITE decoded candidates 中按 LWE prior 选择，不使用 true secret
    - candidate-any: candidate set 中是否存在正确候选，使用 true secret，只作诊断上界

这是最耗时的量子模拟扫描（7 个维度 x 100 个种子），复现 Figure 3 并不需要重跑：
data/figure3_performance_comparison/ 下已归档了全部 per-seed 数据。

输出目录（新的一次运行不会覆盖归档数据）：
<ARTIFACT_ROOT>/output/main_varqite_experiment/<run_name>/
    config.json
    raw_per_seed.csv          # 每完成一个 seed 就 append 写入
    summary_by_param.csv      # 每完成一个 n 就更新写入
    worker_errors.jsonl       # 若某些 seed 出错，逐条追加

注意：
    raw_per_seed.csv 是增量写入的；若实验中断，已完成的 seed 不会丢失。
    summary_by_param.csv 在每个 n 完成后重写一次，包含截至当前所有已完成 n 的汇总。
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------
# 避免“多进程 × BLAS 多线程”导致 CPU 过度订阅。
# ----------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

# ----------------------------------------------------------------------
# 项目根目录。当前脚本应放在：
#   <PROJECT_ROOT>/scripts/performance/
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.instance.lwe_instance import generate_small_lwe_instance
from algorithms.lattice.hnf_basis import full_rank_basis_from_hnf
from algorithms.lattice.reduction import reduce_basis
from algorithms.lattice.babai import babai_nearest_plane
from algorithms.local_model.local_hamiltonian import build_diagonal_hamiltonian
from algorithms.recovery.recover_secret import recover_secret_from_lattice_point
from algorithms.solvers.classical.greedy_local_solver import greedy_local_from_babai
from algorithms.solvers.quantum.qite import (
    qite_optimize_local_pyqpanda3,
    CandidateRerankConfig,
    rerank_varqite_candidates,
)


# =========================================================
# 1) 默认配置
# =========================================================
CONFIG = {
    # scaling 扫描范围：n = n_start, n_start+n_step, ..., n_end
    "n_start": 3,
    "n_end": 10,
    "n_step": 1,

    # calibration 得到的固定工作点：m ≈ c_star * n
    "c_star": 1.325,
    "m_rounding": "ceil",   # ceil / round / floor
    "q": 17,

    # seed 范围 [start_seed, end_seed)
    "start_seed": 0,
    "end_seed": 100,

    # LWE 分布参数：
    # secret_weight_total = P(s_i != 0)，noise_error_rate = P(e_i != 0)
    # 若 generate_small_lwe_instance 支持 secret_probs / noise_probs，会自动传入。
    "secret_weight_total": 0.5,
    "noise_error_rate": 0.25,
    "require_nonzero_secret": True,

    # reduction 配置
    "reduction_method": "lll",
    "delta_lll": 0.75,

    # local space
    "local_values": [-1, 0, 1],

    # classical local baseline (greedy energy descent)
    "greedy_max_steps": None,
    "greedy_tol": 1e-12,

    # VarQITE 配置
    "penalty": 10.0,
    "layers": 1,
    "dtau": 0.02,
    "steps": 5,
    "shots_A_C": 100,
    "shots_E": 100,
    "shots_decode": 500,
    "reg_floor": 0.1,
    "max_delta_norm": 0.05,
    "verbose_every": 1000,
    "auto_calibrate": False,
    "log_qite_steps": False,

    # candidate generation + LWE-prior reranking
    "candidate_topk": 20,
    "candidate_selection": "energy_lwe_prior",  # energy_lwe_prior / lwe_prior / ...
    "expected_secret_weight_ratio": 0.5,
    "expected_noise_rate": 0.25,
    "prior_weight_secret_violation": 1000.0,
    "prior_weight_noise_violation": 1000.0,
    "prior_weight_secret_weight_gap": 20.0,
    "prior_weight_noise_weight_gap": 10.0,
    "prior_weight_noise_l2": 1.0,
    "prior_weight_local_energy": 0.05,
    "prior_weight_count_log": 0.1,
    "candidate_energy_match_tol": 1e-9,
    "decoded_top_states_to_store": 10,

    # 数值比较容差
    "energy_match_tol": 1e-12,

    # 并行配置
    "max_workers": 5,
    "parallel_chunksize": 2,
    "mp_start_method": "spawn",

    # 增量写入配置
    "flush_each_seed": True,

    # 输出 tag
    "tag": "scaling_candidate_lwe_prior_parallel",
}


# =========================================================
# 2) 数据结构
# =========================================================
@dataclass
class RawPerSeedRow:
    # parameter setting
    n: int
    m: int
    q: int
    seed: int

    # true instance
    s_true_centered: list[int]
    s_true_mod_q: list[int]
    e_true: list[int]

    # Babai
    w_babai: list[int]
    residual: list[int]
    initial_residual_energy: float
    s_hat_babai: list[int]
    babai_secret_success: bool

    # exact local energy oracle
    delta_exact: list[int]
    exact_best_energy: float
    w_exact: list[int]
    s_hat_exact: list[int]
    exact_secret_success: bool

    # classical local baseline: greedy energy descent
    delta_classical: list[int]
    classical_best_energy: float
    classical_num_steps: int
    w_classical: list[int]
    s_hat_classical: list[int]
    classical_secret_success: bool

    # primary VarQITE output
    delta_qite_primary: list[int]
    qite_primary_best_energy: float
    w_qite_primary: list[int]
    s_hat_qite_primary: list[int]
    qite_primary_secret_success: bool

    # VarQITE candidate-best after LWE-prior reranking
    delta_qite_candidate_best: list[int]
    qite_candidate_best_energy: float
    qite_candidate_best_count: int
    qite_candidate_best_prob: float
    qite_candidate_best_lwe_prior_score: float
    qite_candidate_best_secret_ternary_violation: int
    qite_candidate_best_noise_ternary_violation: int
    qite_candidate_best_secret_weight: int
    qite_candidate_best_noise_weight: int
    w_qite_candidate_best: list[int]
    s_hat_qite_candidate_best: list[int]
    qite_candidate_best_secret_success: bool

    # candidate-set diagnostics
    qite_candidate_num_unique: int
    qite_candidate_topk_used: int
    qite_candidate_any_secret_success: bool
    qite_candidate_any_geometry_improved: bool
    qite_candidate_any_matches_exact_state: bool
    qite_candidate_any_matches_exact_energy: bool
    min_dist2_true_among_candidates: int | None

    # comparison flags
    exact_improvable_energy: bool
    exact_improvable_state: bool
    babai_fail_but_oracle_fix: bool
    babai_fail_but_classical_fix: bool
    babai_fail_but_qite_primary_fix: bool
    babai_fail_but_qite_candidate_best_fix: bool
    babai_fail_but_qite_candidate_any_fix: bool

    classical_matches_exact_state: bool
    classical_matches_exact_energy: bool
    qite_primary_matches_exact_state: bool
    qite_primary_matches_exact_energy: bool
    qite_candidate_best_matches_exact_state: bool
    qite_candidate_best_matches_exact_energy: bool

    classical_geometry_improved: bool
    qite_primary_geometry_improved: bool
    qite_candidate_best_geometry_improved: bool

    # geometry
    w_true: list[int]
    dist2_true_babai: int
    dist2_true_exact: int
    dist2_true_classical: int
    dist2_true_qite_primary: int
    dist2_true_qite_candidate_best: int

    # timing/resources
    elapsed_total_sec: float
    elapsed_classical_sec: float
    elapsed_qite_sec: float
    elapsed_candidate_sec: float
    qite_num_qubits: int
    qite_num_params: int

    # optional debug
    decoded_top_states: list[tuple[str, int]]
    qite_history_len: int


# =========================================================
# 3) 工具函数
# =========================================================
def vec_to_list(x: np.ndarray) -> list[int]:
    return [int(v) for v in np.asarray(x).reshape(-1)]


def normalize_lattice_vector(x: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(x)).astype(int)


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


def round_m_from_ratio(n: int, c_star: float, mode: str) -> int:
    x = float(c_star) * int(n)
    if mode == "ceil":
        return int(math.ceil(x))
    if mode == "floor":
        return int(math.floor(x))
    if mode == "round":
        return int(round(x))
    raise ValueError(f"Unsupported m_rounding mode: {mode}")


def build_param_list_from_scaling_cfg(cfg: dict[str, Any]) -> list[dict[str, int]]:
    n_start = int(cfg["n_start"])
    n_end = int(cfg["n_end"])
    n_step = int(cfg["n_step"])
    if n_step <= 0:
        raise ValueError("n_step must be positive.")
    if n_end < n_start:
        raise ValueError("n_end must be >= n_start.")

    params = []
    for n in range(n_start, n_end + 1, n_step):
        params.append(
            {
                "n": int(n),
                "m": round_m_from_ratio(n, float(cfg["c_star"]), str(cfg["m_rounding"])),
                "q": int(cfg["q"]),
            }
        )
    return params


def sanitize_filename_part(x: Any) -> str:
    s = str(x).strip()
    if not s:
        return "untagged"
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in s)


def make_run_name(cfg: dict[str, Any], params: list[dict[str, int]], timestamp: str) -> str:
    c_tag = str(float(cfg["c_star"])).replace(".", "p")
    first_m = params[0]["m"] if params else -1
    last_m = params[-1]["m"] if params else -1
    return (
        f"{sanitize_filename_part(cfg.get('tag', 'exp'))}"
        f"_{sanitize_filename_part(cfg.get('candidate_selection', 'energy_lwe_prior'))}"
        f"_n{cfg['n_start']}-{cfg['n_end']}_step{cfg['n_step']}"
        f"_c{c_tag}_{cfg['m_rounding']}"
        f"_m{first_m}-{last_m}"
        f"_q{cfg['q']}"
        f"_s{cfg['start_seed']}-{cfg['end_seed']}"
        f"_layer{cfg['layers']}"
        f"_cand{cfg['candidate_topk']}"
        f"_{timestamp}"
    )


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


def safe_generate_instance(n: int, m: int, q: int, seed: int, cfg: dict[str, Any]):
    """Generate LWE instance while staying compatible with older signatures."""
    sig = inspect.signature(generate_small_lwe_instance)
    kwargs = {"n": n, "m": m, "q": q, "seed": seed}

    if "secret_probs" in sig.parameters:
        kwargs["secret_probs"] = make_secret_probs(float(cfg["secret_weight_total"]))
    if "noise_probs" in sig.parameters:
        kwargs["noise_probs"] = make_noise_probs(float(cfg["noise_error_rate"]))
    if "secret_mode" in sig.parameters and "secret_mode" in cfg:
        kwargs["secret_mode"] = cfg["secret_mode"]
    if "noise_mode" in sig.parameters and "noise_mode" in cfg:
        kwargs["noise_mode"] = cfg["noise_mode"]
    if "require_nonzero_secret" in sig.parameters:
        kwargs["require_nonzero_secret"] = cfg["require_nonzero_secret"]

    inst = generate_small_lwe_instance(**kwargs)

    if cfg.get("require_nonzero_secret", False):
        s_centered = np.asarray(inst.s_centered)
        if np.all(s_centered == 0):
            return None
    return inst


def make_candidate_config(cfg: dict[str, Any]) -> CandidateRerankConfig:
    return CandidateRerankConfig(
        candidate_topk=int(cfg["candidate_topk"]),
        selection_mode=str(cfg["candidate_selection"]),
        expected_secret_weight_ratio=float(cfg["expected_secret_weight_ratio"]),
        expected_noise_rate=float(cfg["expected_noise_rate"]),
        prior_weight_secret_violation=float(cfg["prior_weight_secret_violation"]),
        prior_weight_noise_violation=float(cfg["prior_weight_noise_violation"]),
        prior_weight_secret_weight_gap=float(cfg["prior_weight_secret_weight_gap"]),
        prior_weight_noise_weight_gap=float(cfg["prior_weight_noise_weight_gap"]),
        prior_weight_noise_l2=float(cfg["prior_weight_noise_l2"]),
        prior_weight_local_energy=float(cfg["prior_weight_local_energy"]),
        prior_weight_count_log=float(cfg["prior_weight_count_log"]),
        energy_match_tol=float(cfg["candidate_energy_match_tol"]),
        decoded_top_states_to_store=int(cfg["decoded_top_states_to_store"]),
    )


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(item), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# =========================================================
# 4) 单 seed 求解
# =========================================================
def solve_one_seed_all_methods(n: int, m: int, q: int, seed: int, cfg: dict[str, Any]) -> RawPerSeedRow | None:
    t0_total = time.perf_counter()

    inst = safe_generate_instance(n=n, m=m, q=q, seed=seed, cfg=cfg)
    if inst is None:
        return None

    # 1) q-ary lattice basis + reduction
    B, _, _ = full_rank_basis_from_hnf(inst.A, q)
    D = reduce_basis(B, method=cfg["reduction_method"], delta=cfg["delta_lll"])

    # 2) Babai
    w_babai, _ = babai_nearest_plane(D, inst.c_centered)
    w_babai = normalize_lattice_vector(w_babai)
    residual = np.asarray(inst.c_centered, dtype=int) - w_babai
    initial_residual_energy = float(np.dot(residual, residual))

    s_hat_babai = recover_secret_from_lattice_point(inst.A, w_babai, q)
    babai_secret_success = bool(np.array_equal(s_hat_babai % q, inst.s_mod_q % q))

    # 3) exact local energy oracle
    states, energies = build_diagonal_hamiltonian(
        D=D,
        residual=residual,
        values=tuple(cfg["local_values"]),
    )
    idx_best = int(np.argmin(energies))
    delta_exact = np.asarray(states[idx_best], dtype=int)
    exact_best_energy = float(energies[idx_best])
    w_exact = normalize_lattice_vector(w_babai + D @ delta_exact)
    s_hat_exact = recover_secret_from_lattice_point(inst.A, w_exact, q)
    exact_secret_success = bool(np.array_equal(s_hat_exact % q, inst.s_mod_q % q))

    # 4) classical local baseline: greedy energy descent
    t0_classical = time.perf_counter()
    classical_res = greedy_local_from_babai(
        D=D,
        residual=residual,
        values=tuple(cfg["local_values"]),
        max_steps=cfg["greedy_max_steps"],
        tol=cfg["greedy_tol"],
    )
    elapsed_classical_sec = float(time.perf_counter() - t0_classical)

    delta_classical = np.asarray(classical_res.delta_greedy, dtype=int)
    classical_best_energy = float(classical_res.best_energy)
    w_classical = normalize_lattice_vector(w_babai + D @ delta_classical)
    s_hat_classical = recover_secret_from_lattice_point(inst.A, w_classical, q)
    classical_secret_success = bool(np.array_equal(s_hat_classical % q, inst.s_mod_q % q))

    # 5) local VarQITE primary run
    t0_qite = time.perf_counter()
    qite_res = qite_optimize_local_pyqpanda3(
        D=D,
        residual=residual,
        penalty=cfg["penalty"],
        layers=cfg["layers"],
        dtau=cfg["dtau"],
        steps=cfg["steps"],
        shots_A_C=cfg["shots_A_C"],
        shots_E=cfg["shots_E"],
        shots_decode=cfg["shots_decode"],
        reg_floor=cfg["reg_floor"],
        max_delta_norm=cfg["max_delta_norm"],
        verbose_every=cfg["verbose_every"],
        auto_calibrate=cfg["auto_calibrate"],
        log_steps=cfg["log_qite_steps"],
    )
    elapsed_qite_sec = float(time.perf_counter() - t0_qite)

    # 6) candidate generation + LWE-prior reranking
    t0_candidate = time.perf_counter()
    candidate_config = make_candidate_config(cfg)
    candidate_res = rerank_varqite_candidates(
        qite_result=qite_res,
        inst=inst,
        D=D,
        w_babai=w_babai,
        residual=residual,
        q=q,
        config=candidate_config,
        delta_exact=delta_exact,
        exact_best_energy=exact_best_energy,
    )
    elapsed_candidate_sec = float(time.perf_counter() - t0_candidate)

    primary = candidate_res.primary_candidate
    best = candidate_res.best_candidate

    delta_qite_primary = np.asarray(primary["delta"], dtype=int)
    w_qite_primary = normalize_lattice_vector(np.asarray(primary["w_candidate"], dtype=int))
    s_hat_qite_primary = np.asarray(primary["s_hat"], dtype=int)
    qite_primary_best_energy = float(getattr(qite_res, "best_energy", primary["energy"]))
    qite_primary_secret_success = bool(primary["secret_success"])

    delta_qite_best = np.asarray(best["delta"], dtype=int)
    w_qite_best = normalize_lattice_vector(np.asarray(best["w_candidate"], dtype=int))
    s_hat_qite_best = np.asarray(best["s_hat"], dtype=int)
    qite_candidate_best_secret_success = bool(best["secret_success"])

    # 7) comparison flags
    exact_improvable_energy = bool(exact_best_energy < initial_residual_energy - float(cfg["energy_match_tol"]))
    exact_improvable_state = bool(np.any(delta_exact != 0))

    babai_fail_but_oracle_fix = bool((not babai_secret_success) and exact_secret_success)
    babai_fail_but_classical_fix = bool((not babai_secret_success) and classical_secret_success)
    babai_fail_but_qite_primary_fix = bool((not babai_secret_success) and qite_primary_secret_success)
    babai_fail_but_qite_candidate_best_fix = bool((not babai_secret_success) and qite_candidate_best_secret_success)
    babai_fail_but_qite_candidate_any_fix = bool((not babai_secret_success) and candidate_res.any_secret_success)

    classical_matches_exact_state = bool(np.array_equal(delta_classical, delta_exact))
    classical_matches_exact_energy = bool(abs(classical_best_energy - exact_best_energy) <= cfg["energy_match_tol"])
    qite_primary_matches_exact_state = bool(primary["matches_exact_state"])
    qite_primary_matches_exact_energy = bool(primary["matches_exact_energy"])
    qite_candidate_best_matches_exact_state = bool(best["matches_exact_state"])
    qite_candidate_best_matches_exact_energy = bool(best["matches_exact_energy"])

    # 8) geometry
    w_true = np.asarray(inst.w_true, dtype=int)
    dist2_true_babai = int(np.sum((w_babai - w_true) ** 2))
    dist2_true_exact = int(np.sum((w_exact - w_true) ** 2))
    dist2_true_classical = int(np.sum((w_classical - w_true) ** 2))
    dist2_true_qite_primary = int(primary["dist2_true"])
    dist2_true_qite_candidate_best = int(best["dist2_true"])

    classical_geometry_improved = bool(dist2_true_classical < dist2_true_babai)
    qite_primary_geometry_improved = bool(dist2_true_qite_primary < dist2_true_babai)
    qite_candidate_best_geometry_improved = bool(dist2_true_qite_candidate_best < dist2_true_babai)

    elapsed_total_sec = float(time.perf_counter() - t0_total)

    return RawPerSeedRow(
        n=int(n),
        m=int(m),
        q=int(q),
        seed=int(seed),

        s_true_centered=vec_to_list(inst.s_centered),
        s_true_mod_q=vec_to_list(inst.s_mod_q),
        e_true=vec_to_list(inst.e),

        w_babai=vec_to_list(w_babai),
        residual=vec_to_list(residual),
        initial_residual_energy=float(initial_residual_energy),
        s_hat_babai=vec_to_list(s_hat_babai),
        babai_secret_success=bool(babai_secret_success),

        delta_exact=vec_to_list(delta_exact),
        exact_best_energy=float(exact_best_energy),
        w_exact=vec_to_list(w_exact),
        s_hat_exact=vec_to_list(s_hat_exact),
        exact_secret_success=bool(exact_secret_success),

        delta_classical=vec_to_list(delta_classical),
        classical_best_energy=float(classical_best_energy),
        classical_num_steps=int(classical_res.num_steps),
        w_classical=vec_to_list(w_classical),
        s_hat_classical=vec_to_list(s_hat_classical),
        classical_secret_success=bool(classical_secret_success),

        delta_qite_primary=vec_to_list(delta_qite_primary),
        qite_primary_best_energy=float(qite_primary_best_energy),
        w_qite_primary=vec_to_list(w_qite_primary),
        s_hat_qite_primary=vec_to_list(s_hat_qite_primary),
        qite_primary_secret_success=bool(qite_primary_secret_success),

        delta_qite_candidate_best=vec_to_list(delta_qite_best),
        qite_candidate_best_energy=float(best["energy"]),
        qite_candidate_best_count=int(best["count"]),
        qite_candidate_best_prob=float(best["prob"]),
        qite_candidate_best_lwe_prior_score=float(best["lwe_prior_score"]),
        qite_candidate_best_secret_ternary_violation=int(best["secret_ternary_violation"]),
        qite_candidate_best_noise_ternary_violation=int(best["noise_ternary_violation"]),
        qite_candidate_best_secret_weight=int(best["secret_weight"]),
        qite_candidate_best_noise_weight=int(best["noise_weight"]),
        w_qite_candidate_best=vec_to_list(w_qite_best),
        s_hat_qite_candidate_best=vec_to_list(s_hat_qite_best),
        qite_candidate_best_secret_success=bool(qite_candidate_best_secret_success),

        qite_candidate_num_unique=int(candidate_res.candidate_num_unique),
        qite_candidate_topk_used=int(candidate_res.candidate_topk_used),
        qite_candidate_any_secret_success=bool(candidate_res.any_secret_success),
        qite_candidate_any_geometry_improved=bool(candidate_res.any_geometry_improved),
        qite_candidate_any_matches_exact_state=bool(candidate_res.any_matches_exact_state),
        qite_candidate_any_matches_exact_energy=bool(candidate_res.any_matches_exact_energy),
        min_dist2_true_among_candidates=candidate_res.min_dist2_true_among_candidates,

        exact_improvable_energy=bool(exact_improvable_energy),
        exact_improvable_state=bool(exact_improvable_state),
        babai_fail_but_oracle_fix=bool(babai_fail_but_oracle_fix),
        babai_fail_but_classical_fix=bool(babai_fail_but_classical_fix),
        babai_fail_but_qite_primary_fix=bool(babai_fail_but_qite_primary_fix),
        babai_fail_but_qite_candidate_best_fix=bool(babai_fail_but_qite_candidate_best_fix),
        babai_fail_but_qite_candidate_any_fix=bool(babai_fail_but_qite_candidate_any_fix),

        classical_matches_exact_state=bool(classical_matches_exact_state),
        classical_matches_exact_energy=bool(classical_matches_exact_energy),
        qite_primary_matches_exact_state=bool(qite_primary_matches_exact_state),
        qite_primary_matches_exact_energy=bool(qite_primary_matches_exact_energy),
        qite_candidate_best_matches_exact_state=bool(qite_candidate_best_matches_exact_state),
        qite_candidate_best_matches_exact_energy=bool(qite_candidate_best_matches_exact_energy),

        classical_geometry_improved=bool(classical_geometry_improved),
        qite_primary_geometry_improved=bool(qite_primary_geometry_improved),
        qite_candidate_best_geometry_improved=bool(qite_candidate_best_geometry_improved),

        w_true=vec_to_list(w_true),
        dist2_true_babai=int(dist2_true_babai),
        dist2_true_exact=int(dist2_true_exact),
        dist2_true_classical=int(dist2_true_classical),
        dist2_true_qite_primary=int(dist2_true_qite_primary),
        dist2_true_qite_candidate_best=int(dist2_true_qite_candidate_best),

        elapsed_total_sec=float(elapsed_total_sec),
        elapsed_classical_sec=float(elapsed_classical_sec),
        elapsed_qite_sec=float(elapsed_qite_sec),
        elapsed_candidate_sec=float(elapsed_candidate_sec),
        qite_num_qubits=int(qite_res.num_qubits),
        qite_num_params=int(qite_res.num_params),

        decoded_top_states=candidate_res.decoded_top_states,
        qite_history_len=int(len(qite_res.history)),
    )


def solve_one_seed_worker(args: tuple[int, int, int, int, dict[str, Any]]):
    n, m, q, seed, cfg = args
    try:
        row = solve_one_seed_all_methods(n=n, m=m, q=q, seed=seed, cfg=cfg)
        return int(seed), row, None
    except Exception:
        return int(seed), None, traceback.format_exc()


# =========================================================
# 5) summary 聚合
# =========================================================
def summarize_rows(rows: list[RawPerSeedRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[RawPerSeedRow]] = {}
    for row in rows:
        grouped.setdefault((row.n, row.m, row.q), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (n, m, q), group in sorted(grouped.items()):
        num_total = len(group)
        if num_total == 0:
            continue

        failed_group = [r for r in group if not r.babai_secret_success]
        oracle_fixable_group = [r for r in failed_group if r.exact_secret_success]

        def mean_bool(attr: str) -> float:
            return float(np.mean([bool(getattr(r, attr)) for r in group]))

        def mean_num(attr: str) -> float:
            return float(np.mean([float(getattr(r, attr)) for r in group]))

        oracle_fixable_rate = float(np.mean([r.exact_secret_success for r in failed_group])) if failed_group else float("nan")
        classical_given_oracle_fixable = (
            float(np.mean([r.classical_secret_success for r in oracle_fixable_group]))
            if oracle_fixable_group else float("nan")
        )
        qite_candidate_given_oracle_fixable = (
            float(np.mean([r.qite_candidate_best_secret_success for r in oracle_fixable_group]))
            if oracle_fixable_group else float("nan")
        )

        summary_rows.append(
            {
                "n": int(n),
                "m": int(m),
                "q": int(q),
                "num_seeds": int(num_total),

                # main success metrics
                "babai_secret_success_rate": mean_bool("babai_secret_success"),
                "exact_local_energy_oracle_success_rate": mean_bool("exact_secret_success"),
                "classical_local_baseline_success_rate": mean_bool("classical_secret_success"),
                "varqite_primary_secret_success_rate": mean_bool("qite_primary_secret_success"),
                "varqite_candidate_lwe_prior_success_rate": mean_bool("qite_candidate_best_secret_success"),
                "varqite_candidate_any_success_rate": mean_bool("qite_candidate_any_secret_success"),

                # Babai-failure repair metrics
                "babai_fail_but_oracle_fix_rate": mean_bool("babai_fail_but_oracle_fix"),
                "babai_fail_but_classical_fix_rate": mean_bool("babai_fail_but_classical_fix"),
                "babai_fail_but_qite_primary_fix_rate": mean_bool("babai_fail_but_qite_primary_fix"),
                "babai_fail_but_qite_candidate_best_fix_rate": mean_bool("babai_fail_but_qite_candidate_best_fix"),
                "babai_fail_but_qite_candidate_any_fix_rate": mean_bool("babai_fail_but_qite_candidate_any_fix"),

                # exact / local energy diagnostics
                "exact_improvable_energy_rate": mean_bool("exact_improvable_energy"),
                "exact_improvable_state_rate": mean_bool("exact_improvable_state"),
                "oracle_fixable_rate_among_babai_failures": (
                    float(oracle_fixable_rate) if not np.isnan(oracle_fixable_rate) else ""
                ),
                "classical_given_oracle_fixable_success_rate": (
                    float(classical_given_oracle_fixable) if not np.isnan(classical_given_oracle_fixable) else ""
                ),
                "qite_candidate_given_oracle_fixable_success_rate": (
                    float(qite_candidate_given_oracle_fixable) if not np.isnan(qite_candidate_given_oracle_fixable) else ""
                ),

                # matching exact local energy oracle
                "classical_matches_exact_state_rate": mean_bool("classical_matches_exact_state"),
                "classical_matches_exact_energy_rate": mean_bool("classical_matches_exact_energy"),
                "qite_primary_matches_exact_state_rate": mean_bool("qite_primary_matches_exact_state"),
                "qite_primary_matches_exact_energy_rate": mean_bool("qite_primary_matches_exact_energy"),
                "qite_candidate_best_matches_exact_state_rate": mean_bool("qite_candidate_best_matches_exact_state"),
                "qite_candidate_best_matches_exact_energy_rate": mean_bool("qite_candidate_best_matches_exact_energy"),

                # geometry
                "classical_geometry_improved_rate": mean_bool("classical_geometry_improved"),
                "qite_primary_geometry_improved_rate": mean_bool("qite_primary_geometry_improved"),
                "qite_candidate_best_geometry_improved_rate": mean_bool("qite_candidate_best_geometry_improved"),
                "qite_candidate_any_geometry_improved_rate": mean_bool("qite_candidate_any_geometry_improved"),

                "avg_dist2_true_babai": mean_num("dist2_true_babai"),
                "avg_dist2_true_exact": mean_num("dist2_true_exact"),
                "avg_dist2_true_classical": mean_num("dist2_true_classical"),
                "avg_dist2_true_qite_primary": mean_num("dist2_true_qite_primary"),
                "avg_dist2_true_qite_candidate_best": mean_num("dist2_true_qite_candidate_best"),

                # candidate diagnostics
                "avg_candidate_num_unique": mean_num("qite_candidate_num_unique"),
                "avg_candidate_topk_used": mean_num("qite_candidate_topk_used"),
                "avg_candidate_best_lwe_prior_score": mean_num("qite_candidate_best_lwe_prior_score"),
                "avg_candidate_best_energy": mean_num("qite_candidate_best_energy"),

                # timing/resources
                "avg_time_sec": mean_num("elapsed_qite_sec"),
                "avg_total_time_sec": mean_num("elapsed_total_sec"),
                "avg_classical_time_sec": mean_num("elapsed_classical_sec"),
                "avg_qite_time_sec": mean_num("elapsed_qite_sec"),
                "avg_candidate_rerank_time_sec": mean_num("elapsed_candidate_sec"),
                "avg_qite_num_qubits": mean_num("qite_num_qubits"),
                "avg_qite_num_params": mean_num("qite_num_params"),
            }
        )

    return summary_rows


# =========================================================
# 6) 主函数
# =========================================================
def main():
    cfg = dict(CONFIG)
    params = build_param_list_from_scaling_cfg(cfg)
    cfg["param_list_generated"] = params
    cfg["candidate_rerank_config"] = make_candidate_config(cfg).to_dict()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = make_run_name(cfg, params, timestamp)

    results_root_dir = PROJECT_ROOT / "output" / "main_varqite_experiment"
    run_dir = results_root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = run_dir / "raw_per_seed.csv"
    summary_csv_path = run_dir / "summary_by_param.csv"
    config_json_path = run_dir / "config.json"
    errors_jsonl_path = run_dir / "worker_errors.jsonl"

    with config_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(cfg), f, ensure_ascii=False, indent=2)

    max_workers = resolve_max_workers(cfg)
    mp_start_method = str(cfg.get("mp_start_method", "spawn"))
    parallel_chunksize = max(1, int(cfg.get("parallel_chunksize", 1)))

    print("\n============================================================")
    print("Scaling main experiment 11 starts.")
    print(f"Script path        : {Path(__file__).resolve()}")
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Results root       : {results_root_dir}")
    print(f"Run dir            : {run_dir}")
    print(f"Generated params   : {params}")
    print(f"max_workers        : {max_workers}")
    print(f"mp_start_method    : {mp_start_method}")
    print(f"parallel_chunksize : {parallel_chunksize}")
    print("============================================================\n")

    all_rows: list[RawPerSeedRow] = []
    raw_fieldnames = list(RawPerSeedRow.__dataclass_fields__.keys())

    for param in params:
        n = int(param["n"])
        m = int(param["m"])
        q = int(param["q"])
        print(f"\n=== running scaling point (n={n}, m={m}, q={q}) ===")

        seed_args = [
            (n, m, q, seed, cfg)
            for seed in range(int(cfg["start_seed"]), int(cfg["end_seed"]))
        ]

        point_rows: list[RawPerSeedRow] = []
        point_t0 = time.perf_counter()
        completed = 0
        errors = 0

        ctx = mp.get_context(mp_start_method)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = [executor.submit(solve_one_seed_worker, args) for args in seed_args]
            for fut in as_completed(futures):
                seed, row, error = fut.result()
                completed += 1

                if error is not None:
                    errors += 1
                    append_jsonl(errors_jsonl_path, {"n": n, "m": m, "q": q, "seed": seed, "error": error})
                elif row is not None:
                    point_rows.append(row)
                    all_rows.append(row)
                    append_csv_row(raw_csv_path, asdict(row), raw_fieldnames)

                if completed % 5 == 0 or completed == len(seed_args):
                    print(
                        f"[progress n={n}] completed={completed}/{len(seed_args)}, "
                        f"kept_point={len(point_rows)}, errors_point={errors}, last_seed={seed}"
                    )

        point_rows.sort(key=lambda r: int(r.seed))
        all_rows.sort(key=lambda r: (int(r.n), int(r.m), int(r.q), int(r.seed)))

        summary_rows = summarize_rows(all_rows)
        write_csv(summary_csv_path, summary_rows)

        point_elapsed = time.perf_counter() - point_t0
        print(f"[done n={n}] valid={len(point_rows)}, errors={errors}, elapsed={point_elapsed:.2f} sec")
        print(f"[updated] raw     : {raw_csv_path}")
        print(f"[updated] summary : {summary_csv_path}")

    print("\n============================================================")
    print("Done.")
    print(f"Run dir       : {run_dir}")
    print(f"Config        : {config_json_path}")
    print(f"Raw CSV       : {raw_csv_path}")
    print(f"Summary CSV   : {summary_csv_path}")
    print(f"Errors JSONL  : {errors_jsonl_path}")
    print("============================================================\n")

    for row in summarize_rows(all_rows):
        print(row)


if __name__ == "__main__":
    main()
