from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np

from ..common.pauli import (
    PauliSum,
    cleanup_pauli_sum,
    identity_pstr,
    single_pauli_pstr,
    two_pauli_pstr,
    pauli_string_mul,
    pauli_sum_mul,
)
from ..common.measurement import batch_estimate_pauli_strings, estimate_pauli_sum


def conjugate_pauli_string_by_ry_single(pstr: str, q: int, theta: float) -> PauliSum:
    ch = pstr[q]
    if ch in ("I", "Y"):
        return {pstr: 1.0}
    elif ch == "X":
        s1 = list(pstr)
        s1[q] = "X"
        s2 = list(pstr)
        s2[q] = "Z"
        return {
            "".join(s1): np.cos(theta),
            "".join(s2): -np.sin(theta),
        }
    elif ch == "Z":
        s1 = list(pstr)
        s1[q] = "Z"
        s2 = list(pstr)
        s2[q] = "X"
        return {
            "".join(s1): np.cos(theta),
            "".join(s2): np.sin(theta),
        }
    else:
        raise ValueError(f"bad Pauli char: {ch}")


def conjugate_pauli_sum_by_ry(ps: PauliSum, q: int, theta: float) -> PauliSum:
    out = defaultdict(complex)
    for p, c in ps.items():
        pieces = conjugate_pauli_string_by_ry_single(p, q, theta)
        for pp, cc in pieces.items():
            out[pp] += c * cc
    return cleanup_pauli_sum(dict(out))


def _embed_local_image(num_qubits: int, q: int, ch: str, cnot_c: int, cnot_t: int) -> str:
    if q == cnot_c:
        if ch == "I":
            return identity_pstr(num_qubits)
        if ch == "X":
            return two_pauli_pstr(num_qubits, cnot_c, "X", cnot_t, "X")
        if ch == "Y":
            return two_pauli_pstr(num_qubits, cnot_c, "Y", cnot_t, "X")
        if ch == "Z":
            return single_pauli_pstr(num_qubits, cnot_c, "Z")
    elif q == cnot_t:
        if ch == "I":
            return identity_pstr(num_qubits)
        if ch == "X":
            return single_pauli_pstr(num_qubits, cnot_t, "X")
        if ch == "Y":
            return two_pauli_pstr(num_qubits, cnot_c, "Z", cnot_t, "Y")
        if ch == "Z":
            return two_pauli_pstr(num_qubits, cnot_c, "Z", cnot_t, "Z")
    else:
        return single_pauli_pstr(num_qubits, q, ch) if ch != "I" else identity_pstr(num_qubits)

    raise ValueError(f"bad Pauli char: {ch}")


def conjugate_pauli_string_by_cnot(pstr: str, c: int, t: int):
    num_qubits = len(pstr)
    phase = 1 + 0j
    out = identity_pstr(num_qubits)

    for q, ch in enumerate(pstr):
        img = _embed_local_image(num_qubits, q, ch, c, t)
        ph, out = pauli_string_mul(out, img)
        phase *= ph

    return phase, out


def conjugate_pauli_sum_by_cnot(ps: PauliSum, c: int, t: int) -> PauliSum:
    out = defaultdict(complex)
    for p, coeff in ps.items():
        ph, pp = conjugate_pauli_string_by_cnot(p, c, t)
        out[pp] += coeff * ph
    return cleanup_pauli_sum(dict(out))


def build_generators_from_schedule(
    num_qubits: int,
    schedule: List[tuple],
    params: np.ndarray,
) -> List[PauliSum]:
    param_gate_positions = []
    for pos, op in enumerate(schedule):
        if op[0] == "RY":
            param_gate_positions.append(pos)

    generators: List[PauliSum] = []

    for gate_pos in param_gate_positions:
        _, q, _ = schedule[gate_pos]
        g = {single_pauli_pstr(num_qubits, q, "Y"): 1.0}

        for later_op in schedule[gate_pos + 1:]:
            if later_op[0] == "RY":
                _, q2, idx2 = later_op
                g = conjugate_pauli_sum_by_ry(g, q2, float(params[idx2]))
            elif later_op[0] == "CNOT":
                _, c, t = later_op
                g = conjugate_pauli_sum_by_cnot(g, c, t)
            else:
                raise ValueError(f"unknown op in schedule: {later_op}")

        generators.append(g)

    return generators


def build_A_C_by_measurement(
    vqc,
    params: np.ndarray,
    schedule: List[tuple],
    h_pauli: PauliSum,
    shots: int,
    reg_floor: float = 1e-2,
    bit_reversed: bool = False,
    y_rot_sign: int = 1,
    qvm=None,
):
    """
    通过测量构造 McLachlan VarQITE 的
        A_ij = 1/4 Re <G_i G_j>
        C_i  = 1/2 Im <G_i H>

    关键优化：
    1) A 只算上三角，再镜像
    2) 所有单串期望值一次性按 basis 分组批量测量
    """
    num_qubits = len(next(iter(h_pauli)))
    generators = build_generators_from_schedule(num_qubits, schedule, params)
    num_params = len(generators)

    gg_products = [[None] * num_params for _ in range(num_params)]
    gh_products = [None] * num_params
    needed = set()

    # 只构造 A 的上三角
    for i in range(num_params):
        for j in range(i, num_params):
            ps = pauli_sum_mul(generators[i], generators[j])
            gg_products[i][j] = ps
            needed.update(ps.keys())

        psh = pauli_sum_mul(generators[i], h_pauli)
        gh_products[i] = psh
        needed.update(psh.keys())

    # 一次性按 basis 分组测完所有需要的单串
    cache = batch_estimate_pauli_strings(
        vqc=vqc,
        params=params,
        pstrs=needed,
        shots=shots,
        bit_reversed=bit_reversed,
        y_rot_sign=y_rot_sign,
        qvm=qvm,
    )

    A = np.zeros((num_params, num_params), dtype=np.float64)
    C = np.zeros(num_params, dtype=np.float64)

    for i in range(num_params):
        for j in range(i, num_params):
            val = estimate_pauli_sum(
                vqc=vqc,
                params=params,
                ps=gg_products[i][j],
                shots=shots,
                cache=cache,
                bit_reversed=bit_reversed,
                y_rot_sign=y_rot_sign,
                qvm=qvm,
            )
            Aij = 0.25 * np.real(val)
            A[i, j] = Aij
            A[j, i] = Aij

        val = estimate_pauli_sum(
            vqc=vqc,
            params=params,
            ps=gh_products[i],
            shots=shots,
            cache=cache,
            bit_reversed=bit_reversed,
            y_rot_sign=y_rot_sign,
            qvm=qvm,
        )
        C[i] = 0.5 * np.imag(val)

    A = 0.5 * (A + A.T)

    eigvals = np.linalg.eigvalsh(A)
    lam_min = float(np.min(eigvals))
    if lam_min < reg_floor:
        A = A + (reg_floor - lam_min) * np.eye(A.shape[0])

    return A, C