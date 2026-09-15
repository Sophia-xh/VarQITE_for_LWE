# babai.py
from __future__ import annotations

import numpy as np

from algorithms.lattice.gram_schmidt import gram_schmidt_columns


def babai_nearest_plane(B: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Babai 最近平面算法（列基版本）

    输入:
    - B:      m x n 基矩阵，列是基向量
    - target: R^m 中目标向量

    输出:
    - w:      Babai 输出的近似最近格点
    - coeffs: 该格点在基 B 下的整数坐标
    """
    B = np.array(B, dtype=int)
    t = np.array(target, dtype=float).copy()

    _, n = B.shape
    coeffs = np.zeros(n, dtype=int)

    B_star, _, sqnorm = gram_schmidt_columns(B)

    for j in range(n - 1, -1, -1):
        cj = np.dot(t, B_star[:, j]) / sqnorm[j]
        uj = int(np.round(cj))
        coeffs[j] = uj
        t = t - uj * B[:, j]

    w = B @ coeffs
    return w.astype(int), coeffs