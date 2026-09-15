from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

PauliSum = Dict[str, complex]

PAULI_MUL_1Q = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),

    ("X", "I"): (1, "X"),
    ("X", "X"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),

    ("Y", "I"): (1, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1, "I"),
    ("Y", "Z"): (1j, "X"),

    ("Z", "I"): (1, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1, "I"),
}

I2 = np.eye(2, dtype=np.complex128)
XMAT = np.array([[0, 1], [1, 0]], dtype=np.complex128)
YMAT = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
ZMAT = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULI_MAT_1Q = {
    "I": I2,
    "X": XMAT,
    "Y": YMAT,
    "Z": ZMAT,
}


def cleanup_pauli_sum(ps: PauliSum, tol: float = 1e-12) -> PauliSum:
    out: PauliSum = {}
    for k, v in ps.items():
        if abs(v) > tol:
            out[k] = complex(v)
    return out


def identity_pstr(num_qubits: int) -> str:
    return "I" * num_qubits


def single_pauli_pstr(num_qubits: int, q: int, ch: str) -> str:
    s = ["I"] * num_qubits
    s[q] = ch
    return "".join(s)


def two_pauli_pstr(num_qubits: int, q1: int, ch1: str, q2: int, ch2: str) -> str:
    s = ["I"] * num_qubits
    s[q1] = ch1
    s[q2] = ch2
    return "".join(s)


def pauli_string_mul(p1: str, p2: str) -> Tuple[complex, str]:
    phase = 1 + 0j
    out = []
    for a, b in zip(p1, p2):
        ph, c = PAULI_MUL_1Q[(a, b)]
        phase *= ph
        out.append(c)
    return phase, "".join(out)


def pauli_sum_mul(a: PauliSum, b: PauliSum) -> PauliSum:
    out = defaultdict(complex)
    for p1, c1 in a.items():
        for p2, c2 in b.items():
            ph, p3 = pauli_string_mul(p1, p2)
            out[p3] += c1 * c2 * ph
    return cleanup_pauli_sum(dict(out))


def pauli_string_to_matrix(pstr: str) -> np.ndarray:
    """
    pstr[0] 对应 q0，因此 numpy kron 时用 reversed(pstr)
    保持和测量时的 qubit 编号一致。
    """
    out = np.array([[1]], dtype=np.complex128)
    for ch in reversed(pstr):
        out = np.kron(out, PAULI_MAT_1Q[ch])
    return out