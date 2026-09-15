from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from algorithms.local_model.local_hamiltonian import local_cost


def decode_2bit_ternary_bitstring(bitstr: str, bit_reversed: bool = False):
    """
    2-bit ternary 解码：
        10 -> -1
        00 ->  0
        01 -> +1
        11 -> invalid
    """
    bits = bitstr[::-1] if bit_reversed else bitstr
    num_qubits = len(bits)

    if num_qubits % 2 != 0:
        raise ValueError("2-bit ternary 编码要求 qubit 数必须为偶数。")

    num_vars = num_qubits // 2
    delta = np.zeros(num_vars, dtype=int)

    for i in range(num_vars):
        m = int(bits[2 * i])
        p = int(bits[2 * i + 1])

        if m == 1 and p == 1:
            return False, delta

        delta[i] = p - m

    return True, delta


def pick_best_delta_from_counts(
    counts: Dict[str, int],
    D: np.ndarray,
    residual: np.ndarray,
    bit_reversed: bool = False,
):
    """
    从最终测量 counts 中：
        1) 只保留合法 ternary 状态
        2) 累计每个 delta 的出现次数
        3) 按 local_cost 选最优；若并列则选 count 更大的
    """
    decoded_count_map: Dict[Tuple[int, ...], int] = defaultdict(int)

    for bitstr, cnt in counts.items():
        valid, delta = decode_2bit_ternary_bitstring(bitstr, bit_reversed=bit_reversed)
        if valid:
            decoded_count_map[tuple(int(x) for x in delta)] += int(cnt)

    if not decoded_count_map:
        delta0 = np.zeros(D.shape[1], dtype=int)
        return delta0, {}, local_cost(D, residual, delta0)

    best_delta = None
    best_energy = None
    best_count = None

    for key, cnt in decoded_count_map.items():
        delta = np.array(key, dtype=int)
        e = local_cost(D, residual, delta)

        if (best_energy is None) or (e < best_energy - 1e-12) or (
            abs(e - best_energy) < 1e-12 and cnt > best_count
        ):
            best_delta = delta
            best_energy = e
            best_count = cnt

    return best_delta, dict(decoded_count_map), float(best_energy)