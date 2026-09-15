# gram_schmidt.py
from __future__ import annotations

import numpy as np


def gram_schmidt_columns(B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对列向量基 B=[b1,...,bn] 做 Gram-Schmidt。
    返回:
    - B_star: 正交化后的列向量
    - mu:     GS 系数
    - sqnorm: 每个 b_i^* 的平方范数
    """
    B = np.asarray(B, dtype=float)
    m, n = B.shape

    B_star = np.zeros((m, n), dtype=float)
    mu = np.zeros((n, n), dtype=float)
    sqnorm = np.zeros(n, dtype=float)

    for i in range(n):
        v = B[:, i].copy()
        for j in range(i):
            if sqnorm[j] < 1e-14:
                raise ValueError("Gram-Schmidt 遇到近零向量，基可能退化")
            mu[i, j] = np.dot(B[:, i], B_star[:, j]) / sqnorm[j]
            v -= mu[i, j] * B_star[:, j]

        B_star[:, i] = v
        sqnorm[i] = np.dot(v, v)

    return B_star, mu, sqnorm