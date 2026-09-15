from __future__ import annotations

from typing import List

import numpy as np

from ..common.pauli import pauli_string_to_matrix
from ..common.backend import get_statevector
from ..common.measurement import estimate_pauli_string


def exact_pauli_expectation(vqc, params: np.ndarray, pstr: str) -> float:
    """
    仅用于自动校准时的 exact reference。
    """
    psi = get_statevector(vqc, params)
    P = pauli_string_to_matrix(pstr)
    val = np.vdot(psi, P @ psi)

    if not np.isfinite(np.real(val)) or not np.isfinite(np.imag(val)):
        raise RuntimeError(f"exact_pauli_expectation() 数值异常: pstr={pstr}, val={val}")

    return float(np.real(val))


def calibrate_measurement_convention(
    vqc,
    params: np.ndarray,
    candidate_paulis: List[str],
    shots: int = 8000,
) -> dict:
    """
    自动校准测量约定：
        bit_reversed ∈ {False, True}
        y_rot_sign   ∈ {-1, +1}

    如果 exact/statevector 路径不稳定，则自动退回：
        bit_reversed=True, y_rot_sign=1
    """
    try:
        best = None
        uniq = list(dict.fromkeys(candidate_paulis))

        for bit_reversed in [False, True]:
            for y_rot_sign in [-1, +1]:
                errs = []
                for pstr in uniq:
                    exact = exact_pauli_expectation(vqc, params, pstr)
                    est = estimate_pauli_string(
                        vqc=vqc,
                        params=params,
                        pstr=pstr,
                        shots=shots,
                        bit_reversed=bit_reversed,
                        y_rot_sign=y_rot_sign,
                    )
                    errs.append(abs(est - exact))

                mae = float(np.mean(errs))
                if best is None or mae < best["mae"]:
                    best = {
                        "bit_reversed": bit_reversed,
                        "y_rot_sign": y_rot_sign,
                        "mae": mae,
                    }

        return best

    except Exception as e:
        print("[warn] auto_calibrate failed, fallback to default convention.")
        print(f"[warn] calibration exception: {repr(e)}")
        return {
            "bit_reversed": True,
            "y_rot_sign": 1,
            "mae": float("nan"),
        }