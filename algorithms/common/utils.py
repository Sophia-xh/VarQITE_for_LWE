# utils.py
from __future__ import annotations

import numpy as np


def centered_mod(x: np.ndarray | int, q: int) -> np.ndarray:
    """
    把模 q 的值映射到中心区间 [ -q/2, q/2 ) 的整数代表。
    对奇素数 q 最自然。
    """
    x_arr = np.asarray(x, dtype=int)
    half = q // 2
    return ((x_arr + half) % q) - half


def egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def mod_inv(a: int, q: int) -> int:
    a %= q
    g, x, _ = egcd(a, q)
    if g != 1:
        raise ValueError(f"{a} 在模 {q} 下不可逆")
    return x % q


def rank_mod_prime(A: np.ndarray, q: int) -> int:
    """
    计算矩阵 A 在模 q 下的秩。这里默认 q 为素数。
    """
    M = np.array(A, dtype=int) % q
    m, n = M.shape
    row = 0
    rank = 0

    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r, col] % q != 0:
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]

        inv = mod_inv(int(M[row, col]), q)
        M[row, :] = (M[row, :] * inv) % q

        for r in range(m):
            if r != row and M[r, col] % q != 0:
                factor = int(M[r, col])
                M[r, :] = (M[r, :] - factor * M[row, :]) % q

        row += 1
        rank += 1
        if row == m:
            break

    return rank


def solve_mod_linear_system(A: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """
    求解 A x = b (mod q)
    这里默认 q 是素数，且当前 demo 中 A 选成满列秩。
    若方程有唯一解，会返回该解。
    """
    A = np.array(A, dtype=int) % q
    b = np.array(b, dtype=int).reshape(-1, 1) % q

    m, n = A.shape
    aug = np.hstack([A, b])

    row = 0
    pivots: list[int] = []

    for col in range(n):
        pivot = None
        for r in range(row, m):
            if aug[r, col] % q != 0:
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != row:
            aug[[row, pivot], :] = aug[[pivot, row], :]

        inv = mod_inv(int(aug[row, col]), q)
        aug[row, :] = (aug[row, :] * inv) % q

        for r in range(m):
            if r != row and aug[r, col] % q != 0:
                factor = int(aug[r, col])
                aug[r, :] = (aug[r, :] - factor * aug[row, :]) % q

        pivots.append(col)
        row += 1
        if row == m:
            break

    # 检查一致性
    for r in range(row, m):
        if np.all(aug[r, :-1] % q == 0) and aug[r, -1] % q != 0:
            raise ValueError("模线性系统无解")

    if len(pivots) < n:
        raise ValueError(
            "当前系统在模 q 下不是唯一可解的。为了这个 demo，建议取 m=n 且 A 满秩。"
        )

    x = np.zeros(n, dtype=int)
    for r, c in enumerate(pivots):
        x[c] = int(aug[r, -1]) % q

    return x


def vec_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(np.linalg.norm(x))