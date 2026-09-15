# hnf_basis.py
from __future__ import annotations

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


def qary_generator_matrix(A: np.ndarray, q: int) -> np.ndarray:
    """
    M = [A | q I_m]
    其列生成 q-ary lattice:
        Lambda_q(A) = A Z^n + q Z^m
    """
    A = np.array(A, dtype=int)
    m = A.shape[0]
    return np.hstack([A, q * np.eye(m, dtype=int)])


def _try_extract_from_hnf_of_M(M: np.ndarray, m: int) -> tuple[np.ndarray | None, np.ndarray]:
    """
    先直接对 M 做 HNF。
    如果 sympy 的 HNF 在当前版本下相当于“列风格”，
    那么通常会得到恰好 m 个非零列。
    """
    H = np.array(hermite_normal_form(Matrix(M))).astype(int)

    nonzero_cols = [j for j in range(H.shape[1]) if not np.all(H[:, j] == 0)]
    if len(nonzero_cols) == m:
        B = H[:, nonzero_cols]
        if np.linalg.matrix_rank(B) == m:
            return B, H

    return None, H


def _try_extract_from_hnf_of_MT(M: np.ndarray, m: int) -> tuple[np.ndarray | None, np.ndarray]:
    """
    再尝试对 M^T 做 HNF。
    如果当前 sympy 行为更接近“行风格”，
    那么 HNF(M^T) 可能有恰好 m 个非零行；
    这些行转置后就是列基。
    """
    Ht = np.array(hermite_normal_form(Matrix(M.T))).astype(int)

    nonzero_rows = [i for i in range(Ht.shape[0]) if not np.all(Ht[i, :] == 0)]
    if len(nonzero_rows) == m:
        B = Ht[nonzero_rows, :].T
        if np.linalg.matrix_rank(B) == m:
            return B, Ht.T

    return None, Ht.T


def full_rank_basis_from_hnf(A: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从生成矩阵 M=[A|qI] 中提取一个满秩 m x m 基 B。

    返回:
    - B: 满秩基
    - M: 原始生成矩阵
    - H_like: 用于提取的 HNF 结果（便于调试打印）
    """
    M = qary_generator_matrix(A, q)
    m = A.shape[0]

    # 尝试 1：直接对 M 做 HNF
    B, H_like = _try_extract_from_hnf_of_M(M, m)
    if B is not None:
        return B, M, H_like

    # 尝试 2：对 M^T 做 HNF
    B, H_like = _try_extract_from_hnf_of_MT(M, m)
    if B is not None:
        return B, M, H_like

    raise RuntimeError(
        "无法从 HNF 稳定提取满秩格基。\n"
        f"M.shape = {M.shape}, m = {m}\n"
        "建议把当前报错时打印出来的 M 和 H_like 发给我，我继续帮你改。"
    )