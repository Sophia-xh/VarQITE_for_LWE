# reduction.py
from __future__ import annotations

import numpy as np

from algorithms.lattice.gram_schmidt import gram_schmidt_columns


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """
    一个最小可用的 LLL 约简（列基版本）。
    为了 demo 简洁，每次更新后重新做 GSO。
    """
    B = np.array(B, dtype=int).copy()
    n = B.shape[1]

    k = 1
    while k < n:
        # size reduction
        B_star, mu, sqnorm = gram_schmidt_columns(B)
        for j in range(k - 1, -1, -1):
            q_round = int(np.round(mu[k, j]))
            if q_round != 0:
                B[:, k] -= q_round * B[:, j]

        # Lovasz condition
        B_star, mu, sqnorm = gram_schmidt_columns(B)
        lhs = sqnorm[k]
        rhs = (delta - mu[k, k - 1] ** 2) * sqnorm[k - 1]

        if lhs >= rhs - 1e-12:
            k += 1
        else:
            B[:, [k, k - 1]] = B[:, [k - 1, k]]
            k = max(k - 1, 1)

    return B


def reduce_basis(
    B: np.ndarray,
    method: str = "lll",
    delta: float = 0.75,
    bkz_block_size: int = 10,
) -> np.ndarray:
    """
    当前最小 demo：
    - method='lll': 真的跑 LLL
    - method='bkz': 先回退到 LLL，接口先保留
    """
    method = method.lower()

    if method == "lll":
        return lll_reduce(B, delta=delta)

    if method == "bkz":
        print(
            f"[warn] 当前最小 demo 里 BKZ 先回退到 LLL。"
            f"若你要接 fpylll，我下一步可以直接给你 BKZ 包装。"
            f" (请求 block size = {bkz_block_size})"
        )
        return lll_reduce(B, delta=delta)

    raise ValueError(f"未知约简方法: {method}")