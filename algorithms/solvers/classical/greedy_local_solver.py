# greedy_local_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from algorithms.local_model.local_hamiltonian import local_cost


@dataclass
class GreedyLocalResult:
    """
    经典局部贪心搜索结果。

    字段说明：
    - delta_greedy: 最终找到的局部修正向量
    - best_energy:  最终能量 C(delta)
    - num_steps:    实际执行了多少次“接受改进”的步数
    - history:      每一步的日志，便于后续写 JSON / 调试 / 画轨迹
    """
    delta_greedy: np.ndarray
    best_energy: float
    num_steps: int
    history: list[dict[str, Any]]


def _validate_values(values: tuple[int, ...]) -> tuple[int, ...]:
    """
    检查每个坐标允许的离散取值集合。
    默认应为 (-1, 0, 1)。
    """
    if len(values) == 0:
        raise ValueError("values 不能为空。")
    if len(set(values)) != len(values):
        raise ValueError(f"values 中存在重复元素: {values}")
    return tuple(int(v) for v in values)


def _validate_start_delta(start_delta: np.ndarray, num_vars: int, values: tuple[int, ...]) -> np.ndarray:
    """
    检查初始 delta 是否合法。
    """
    delta = np.asarray(start_delta, dtype=int).reshape(-1)
    if delta.shape[0] != num_vars:
        raise ValueError(f"start_delta 维度不匹配: 期望 {num_vars}, 实际 {delta.shape[0]}")
    allowed = set(values)
    for x in delta:
        if int(x) not in allowed:
            raise ValueError(f"start_delta 中出现非法取值 {x}，允许集合为 {values}")
    return delta


def greedy_local_optimize(
    D: np.ndarray,
    residual: np.ndarray,
    values: tuple[int, ...] = (-1, 0, 1),
    start_delta: np.ndarray | None = None,
    max_steps: int | None = None,
    tol: float = 1e-12,
) -> GreedyLocalResult:
    """
    在 full-local 空间上做 best-improvement greedy local search。

    优化目标：
        C(delta) = || D delta - residual ||^2

    搜索空间：
        delta_i ∈ values, 对所有 i = 1, ..., n

    算法：
    - 从 start_delta 开始（默认从 delta=0，也就是 Babai 点）
    - 每一步穷举所有“单坐标改动”
    - 选择能量下降最多的一步
    - 若没有严格下降，则停止

    参数：
    - D:         局部修正矩阵，形状 (m, n)
    - residual:  Babai 残差，形状 (m,)
    - values:    每个 delta_i 允许取的离散值，默认 (-1,0,1)
    - start_delta:
                 初始点；若为 None，则从全零向量开始
    - max_steps:
                 最多接受多少步改进；None 表示不额外限制
    - tol:
                 判断“严格改进”时的数值容差

    返回：
    - GreedyLocalResult
    """
    D = np.asarray(D, dtype=int)
    residual = np.asarray(residual, dtype=int).reshape(-1)

    if D.ndim != 2:
        raise ValueError(f"D 应为二维矩阵，实际 ndim={D.ndim}")
    if residual.ndim != 1:
        raise ValueError("residual 必须是一维向量。")
    if D.shape[0] != residual.shape[0]:
        raise ValueError(
            f"D 和 residual 形状不匹配: D.shape={D.shape}, residual.shape={residual.shape}"
        )

    values = _validate_values(values)
    num_vars = D.shape[1]

    if start_delta is None:
        delta = np.zeros(num_vars, dtype=int)
    else:
        delta = _validate_start_delta(start_delta, num_vars, values)

    # 当前 diff 与能量：
    # diff = D delta - residual
    diff = D @ delta - residual
    current_energy = float(diff @ diff)

    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "accepted": True,
            "delta": delta.tolist(),
            "energy": current_energy,
            "move_coord": None,
            "old_value": None,
            "new_value": None,
            "energy_drop": 0.0,
        }
    ]

    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        best_energy = current_energy
        best_move: tuple[int, int, np.ndarray] | None = None
        # best_move = (coord, new_value, new_diff)

        # ---------------------------------------------------------
        # best-improvement:
        # 枚举所有单坐标改动，找能量下降最多的一步
        # ---------------------------------------------------------
        for i in range(num_vars):
            old_val = int(delta[i])

            for new_val in values:
                if new_val == old_val:
                    continue

                # 单坐标变化量
                change = int(new_val - old_val)

                # 增量更新：
                # diff_new = diff + D[:, i] * (new_val - old_val)
                cand_diff = diff + D[:, i] * change
                cand_energy = float(cand_diff @ cand_diff)

                # 只接受“严格更优”的候选
                if cand_energy < best_energy - tol:
                    best_energy = cand_energy
                    best_move = (i, int(new_val), cand_diff.copy())

        # 没有任何严格下降，停止
        if best_move is None:
            break

        # 接受最优单步
        i_best, new_val_best, diff_best = best_move
        old_val_best = int(delta[i_best])

        delta[i_best] = new_val_best
        diff = diff_best
        energy_drop = current_energy - best_energy
        current_energy = best_energy
        step += 1

        history.append(
            {
                "step": step,
                "accepted": True,
                "delta": delta.tolist(),
                "energy": current_energy,
                "move_coord": int(i_best),
                "old_value": old_val_best,
                "new_value": int(new_val_best),
                "energy_drop": float(energy_drop),
            }
        )

    return GreedyLocalResult(
        delta_greedy=delta.copy(),
        best_energy=float(current_energy),
        num_steps=int(step),
        history=history,
    )


def greedy_local_from_babai(
    D: np.ndarray,
    residual: np.ndarray,
    values: tuple[int, ...] = (-1, 0, 1),
    max_steps: int | None = None,
    tol: float = 1e-12,
) -> GreedyLocalResult:
    """
    一个更直观的包装：
    直接从 Babai 点对应的 delta=0 开始做贪心局部搜索。
    """
    num_vars = np.asarray(D).shape[1]
    start_delta = np.zeros(num_vars, dtype=int)
    return greedy_local_optimize(
        D=D,
        residual=residual,
        values=values,
        start_delta=start_delta,
        max_steps=max_steps,
        tol=tol,
    )