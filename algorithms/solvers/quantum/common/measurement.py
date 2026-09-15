from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .pauli import PauliSum
from .backend import run_vqc_and_get_counts


BasisKey = Tuple[str, ...]


def _basis_key_from_pstr(pstr: str, y_rot_sign: int = 1) -> BasisKey:
    """
    把一个 Pauli string 映射到“测量基 key”。

    约定：
        I / Z -> Z-basis（不做旋转）
        X     -> H
        Y     -> RX(+pi/2) 或 RX(-pi/2)

    返回的是每个 qubit 的 basis 标记元组，例如：
        ("Z", "X", "Y+", "Z", ...)
    """
    key = []
    for ch in pstr:
        if ch in ("I", "Z"):
            key.append("Z")
        elif ch == "X":
            key.append("X")
        elif ch == "Y":
            key.append("Y+" if y_rot_sign >= 0 else "Y-")
        else:
            raise ValueError(f"非法 Pauli 字符: {ch}")
    return tuple(key)


def _basis_rotations_from_key(key: BasisKey) -> List[Tuple[int, str]]:
    """
    basis key -> backend 所需的 basis_rotations
    """
    basis_rotations: List[Tuple[int, str]] = []
    for q, tag in enumerate(key):
        if tag == "Z":
            pass
        elif tag == "X":
            basis_rotations.append((q, "H"))
        elif tag == "Y+":
            basis_rotations.append((q, "RX+PI/2"))
        elif tag == "Y-":
            basis_rotations.append((q, "RX-PI/2"))
        else:
            raise ValueError(f"未知 basis tag: {tag}")
    return basis_rotations


def _pauli_expectation_from_counts(
    counts: Dict[str, int],
    pstr: str,
    bit_reversed: bool = False,
) -> float:
    """
    从同一份 counts 中计算指定 Pauli string 的期望值。
    这里要求：这份 counts 的测量基已经与该 pstr 兼容。
    """
    if all(ch == "I" for ch in pstr):
        return 1.0

    total = 0.0
    total_shots = sum(counts.values())
    if total_shots <= 0:
        raise RuntimeError("_pauli_expectation_from_counts(): total_shots <= 0")

    for bitstr, cnt in counts.items():
        bits = bitstr[::-1] if bit_reversed else bitstr

        eig = 1.0
        for q, ch in enumerate(pstr):
            if ch != "I":
                eig *= (1.0 if bits[q] == "0" else -1.0)

        total += eig * cnt

    return total / total_shots


def batch_estimate_pauli_strings(
    vqc,
    params: np.ndarray,
    pstrs: Iterable[str],
    shots: int,
    bit_reversed: bool = False,
    y_rot_sign: int = 1,
    qvm=None,
) -> Dict[str, float]:
    """
    关键加速函数：
    对一批 Pauli strings 按 measurement basis 分组，
    每个 basis 只跑一次电路，然后从同一份 counts 中同时计算多条串的期望值。

    返回：
        {pstr: expectation}
    """
    uniq_pstrs = list(dict.fromkeys(pstrs))
    if len(uniq_pstrs) == 0:
        return {}

    out: Dict[str, float] = {}

    # 先把纯 I 的情况剔出来
    nontrivial = []
    for p in uniq_pstrs:
        if all(ch == "I" for ch in p):
            out[p] = 1.0
        else:
            nontrivial.append(p)

    if len(nontrivial) == 0:
        return out

    groups: Dict[BasisKey, List[str]] = defaultdict(list)
    for p in nontrivial:
        key = _basis_key_from_pstr(p, y_rot_sign=y_rot_sign)
        groups[key].append(p)

    for key, plist in groups.items():
        basis_rotations = _basis_rotations_from_key(key)

        counts = run_vqc_and_get_counts(
            vqc=vqc,
            params=params,
            num_qubits=len(plist[0]),
            shots=shots,
            basis_rotations=basis_rotations,
            qvm=qvm,
        )

        for p in plist:
            out[p] = _pauli_expectation_from_counts(
                counts=counts,
                pstr=p,
                bit_reversed=bit_reversed,
            )

    return out


def estimate_pauli_string(
    vqc,
    params: np.ndarray,
    pstr: str,
    shots: int,
    bit_reversed: bool = False,
    y_rot_sign: int = 1,
    qvm=None,
) -> float:
    """
    单条串接口：内部复用 batch 版本
    """
    return batch_estimate_pauli_strings(
        vqc=vqc,
        params=params,
        pstrs=[pstr],
        shots=shots,
        bit_reversed=bit_reversed,
        y_rot_sign=y_rot_sign,
        qvm=qvm,
    )[pstr]


def estimate_pauli_sum(
    vqc,
    params: np.ndarray,
    ps: PauliSum,
    shots: int,
    cache: dict | None = None,
    bit_reversed: bool = False,
    y_rot_sign: int = 1,
    qvm=None,
) -> complex:
    """
    估计一个 PauliSum 的期望值。
    若 cache 不为空，则直接复用 cache 里的单串期望值。
    """
    if cache is None:
        cache = {}

    needed = [pstr for pstr in ps.keys() if pstr not in cache]
    if needed:
        new_vals = batch_estimate_pauli_strings(
            vqc=vqc,
            params=params,
            pstrs=needed,
            shots=shots,
            bit_reversed=bit_reversed,
            y_rot_sign=y_rot_sign,
            qvm=qvm,
        )
        cache.update(new_vals)

    val = 0.0 + 0.0j
    for pstr, coeff in ps.items():
        val += coeff * cache[pstr]
    return val


def measured_energy(
    vqc,
    params: np.ndarray,
    h_pauli: PauliSum,
    shots: int,
    bit_reversed: bool = False,
    y_rot_sign: int = 1,
    qvm=None,
) -> float:
    return float(np.real(
        estimate_pauli_sum(
            vqc=vqc,
            params=params,
            ps=h_pauli,
            shots=shots,
            cache=None,
            bit_reversed=bit_reversed,
            y_rot_sign=y_rot_sign,
            qvm=qvm,
        )
    ))