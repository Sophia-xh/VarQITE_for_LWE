from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import numpy as np

from ..common.pauli import (
    PauliSum,
    cleanup_pauli_sum,
    identity_pstr,
    single_pauli_pstr,
    two_pauli_pstr,
)


def _first_bad_index(x: np.ndarray):
    bad = np.argwhere(~np.isfinite(x))
    if bad.size == 0:
        return None
    return tuple(int(v) for v in bad[0])


def _validate_local_inputs(D: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = np.asarray(D, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)

    if D.ndim != 2:
        raise ValueError(f"D 应为二维矩阵，但收到 shape={D.shape}")
    if residual.ndim != 1:
        raise ValueError(f"residual 应为一维向量，但收到 shape={residual.shape}")
    if D.shape[0] != residual.shape[0]:
        raise ValueError(
            f"D 与 residual 维度不匹配：D.shape={D.shape}, residual.shape={residual.shape}"
        )

    if not np.isfinite(D).all():
        idx = _first_bad_index(D)
        raise ValueError(
            f"build_2bit_ternary_local_hamiltonian: D 中存在非有限值 "
            f"(first_bad_index={idx}, value={D[idx] if idx is not None else 'unknown'})"
        )

    if not np.isfinite(residual).all():
        idx = _first_bad_index(residual)
        raise ValueError(
            f"build_2bit_ternary_local_hamiltonian: residual 中存在非有限值 "
            f"(first_bad_index={idx}, value={residual[idx] if idx is not None else 'unknown'})"
        )

    return D, residual


def _normalize_local_inputs(
    D: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    同时缩放 D 和 residual，不改变
        argmin_delta ||D delta - residual||^2
    的最优 delta，只会整体缩放能量尺度。
    """
    max_abs_D = float(np.max(np.abs(D))) if D.size > 0 else 0.0
    max_abs_r = float(np.max(np.abs(residual))) if residual.size > 0 else 0.0
    scale = max(1.0, max_abs_D, max_abs_r)

    Dn = D / scale
    rn = residual / scale
    return Dn, rn, scale


def _assert_all_finite_named(**kwargs) -> None:
    for name, x in kwargs.items():
        x = np.asarray(x)
        if not np.isfinite(x).all():
            idx = _first_bad_index(x)
            raise ValueError(
                f"build_2bit_ternary_local_hamiltonian: {name} 中存在非有限值 "
                f"(first_bad_index={idx}, value={x[idx] if idx is not None else 'unknown'})"
            )


def build_2bit_ternary_local_hamiltonian(
    D: np.ndarray,
    residual: np.ndarray,
    penalty: float = 10.0,
) -> Tuple[PauliSum, int]:
    """
    把局部代价函数
        C(delta) = || D delta - residual ||^2
    写成 2-bit ternary 编码下的对角 Pauli-Z 哈密顿量。

    编码：
        qubit(2i)   = m_i
        qubit(2i+1) = p_i

        10 -> delta_i = -1
        00 -> delta_i =  0
        01 -> delta_i = +1
        11 -> invalid（加 penalty）

    即：
        delta_i = -n(m_i) + n(p_i)
        invalid penalty = penalty * n(m_i)n(p_i)
        其中 n=(I-Z)/2
    """
    D, residual = _validate_local_inputs(D, residual)
    D, residual, _ = _normalize_local_inputs(D, residual)

    num_vars = D.shape[1]
    num_qubits = 2 * num_vars

    G = np.einsum("ki,kj->ij", D, D, optimize=True)
    G = 0.5 * (G + G.T)

    h = np.einsum("ki,k->i", D, residual, optimize=True)

    _assert_all_finite_named(G=G, h=h)

    # delta = S @ n_bits
    S = np.zeros((num_vars, num_qubits), dtype=np.float64)
    for i in range(num_vars):
        S[i, 2 * i] = -1.0
        S[i, 2 * i + 1] = 1.0

    K = np.einsum("ip,ij,jq->pq", S, G, S, optimize=True)
    K = 0.5 * (K + K.T)

    l = -2.0 * np.einsum("ip,i->p", S, h, optimize=True)

    _assert_all_finite_named(S=S, K=K, l=l)

    linear = np.diag(K) + l
    quad = defaultdict(float)

    for a in range(num_qubits):
        for b in range(a + 1, num_qubits):
            coeff = K[a, b] + K[b, a]
            if abs(coeff) > 1e-12:
                quad[(a, b)] += float(coeff)

    # invalid penalty
    for i in range(num_vars):
        quad[(2 * i, 2 * i + 1)] += float(penalty)

    ps = defaultdict(complex)

    # 常数项 ||residual||^2
    ps[identity_pstr(num_qubits)] += float(residual @ residual)

    # 线性项：n=(I-Z)/2
    for a, c in enumerate(linear):
        if abs(c) < 1e-12:
            continue
        ps[identity_pstr(num_qubits)] += 0.5 * c
        ps[single_pauli_pstr(num_qubits, a, "Z")] += -0.5 * c

    # 二次项：n_a n_b=(I-Z_a-Z_b+Z_a Z_b)/4
    for (a, b), c in quad.items():
        if abs(c) < 1e-12:
            continue
        ps[identity_pstr(num_qubits)] += 0.25 * c
        ps[single_pauli_pstr(num_qubits, a, "Z")] += -0.25 * c
        ps[single_pauli_pstr(num_qubits, b, "Z")] += -0.25 * c
        ps[two_pauli_pstr(num_qubits, a, "Z", b, "Z")] += 0.25 * c

    return cleanup_pauli_sum(dict(ps)), num_qubits