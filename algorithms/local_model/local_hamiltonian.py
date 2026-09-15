# local_hamiltonian.py
from __future__ import annotations

import itertools
import numpy as np


def enumerate_ternary_states(num_vars: int, values=(-1, 0, 1)) -> list[tuple[int, ...]]:
    return list(itertools.product(values, repeat=num_vars))


def local_cost(D: np.ndarray, residual: np.ndarray, delta: np.ndarray) -> float:
    """
    C(delta) = || D delta - residual ||^2

    其中:
    residual = c_bar - w_B
    修正后的格点:
        w(delta) = w_B + D delta
    """
    D = np.asarray(D, dtype=int)
    residual = np.asarray(residual, dtype=int).reshape(-1)
    delta = np.asarray(delta, dtype=int).reshape(-1)

    diff = D @ delta - residual
    return float(diff @ diff)


def build_diagonal_hamiltonian(
    D: np.ndarray,
    residual: np.ndarray,
    values=(-1, 0, 1),
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """
    对小规模 demo，直接枚举所有 ternary delta 状态，
    形成一个“对角哈密顿量”的能量表。
    """
    num_vars = D.shape[1]
    states = enumerate_ternary_states(num_vars, values=values)
    energies = np.array([local_cost(D, residual, np.array(st)) for st in states], dtype=float)
    return states, energies