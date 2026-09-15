from __future__ import annotations

import time
import numpy as np

from ..common.backend import _HAS_PYQPANDA3, _fresh_qvm, _finalize_qvm, run_vqc_and_get_counts
from ..common.measurement import measured_energy
from .types import VarQITEResult
from .encoding import build_2bit_ternary_local_hamiltonian
from .ansatz import build_ry_chain_ansatz_with_biased_init
from .generators import build_A_C_by_measurement
from .decode import pick_best_delta_from_counts
from .calibrate import calibrate_measurement_convention

def _first_bad_index(x: np.ndarray):
    bad = np.argwhere(~np.isfinite(x))
    if bad.size == 0:
        return None
    return tuple(int(v) for v in bad[0])


def _prepare_local_inputs(
    D: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    返回：
        D_orig, residual_orig, D_work, residual_work, scale
    其中 work 版本做了数值归一化，orig 版本保留给最终 decode / 真能量计算。
    """
    D_orig = np.asarray(D, dtype=np.float64)
    residual_orig = np.asarray(residual, dtype=np.float64).reshape(-1)

    if D_orig.ndim != 2:
        raise ValueError(f"D 应为二维矩阵，但收到 shape={D_orig.shape}")
    if residual_orig.ndim != 1:
        raise ValueError(f"residual 应为一维向量，但收到 shape={residual_orig.shape}")
    if D_orig.shape[0] != residual_orig.shape[0]:
        raise ValueError(
            f"D 与 residual 维度不匹配：D.shape={D_orig.shape}, residual.shape={residual_orig.shape}"
        )

    if not np.isfinite(D_orig).all():
        idx = _first_bad_index(D_orig)
        raise ValueError(
            f"qite_optimize_local_pyqpanda3: D 中存在非有限值 "
            f"(first_bad_index={idx}, value={D_orig[idx] if idx is not None else 'unknown'})"
        )
    if not np.isfinite(residual_orig).all():
        idx = _first_bad_index(residual_orig)
        raise ValueError(
            f"qite_optimize_local_pyqpanda3: residual 中存在非有限值 "
            f"(first_bad_index={idx}, value={residual_orig[idx] if idx is not None else 'unknown'})"
        )

    max_abs_D = float(np.max(np.abs(D_orig))) if D_orig.size > 0 else 0.0
    max_abs_r = float(np.max(np.abs(residual_orig))) if residual_orig.size > 0 else 0.0
    scale = max(1.0, max_abs_D, max_abs_r)

    D_work = D_orig / scale
    residual_work = residual_orig / scale

    return D_orig, residual_orig, D_work, residual_work, scale


def qite_optimize_local_pyqpanda3(
    D: np.ndarray,
    residual: np.ndarray,
    penalty: float = 10.0,
    layers: int = 1,
    theta_init: np.ndarray | None = None,
    dtau: float = 0.02,
    steps: int = 80,
    shots_A_C: int = 4000,
    shots_E: int = 2000,
    shots_decode: int = 6000,
    reg_floor: float = 0.1,
    max_delta_norm: float = 0.05,
    verbose_every: int = 10,
    auto_calibrate: bool = False,
    log_steps: bool = True,
) -> VarQITEResult:
    if not _HAS_PYQPANDA3:
        raise ImportError("当前环境没有 pyqpanda3，因此无法运行真实量子线路版 VarQITE。")

    D_orig, residual_orig, D_work, residual_work, _ = _prepare_local_inputs(D, residual)

    h_pauli, num_qubits = build_2bit_ternary_local_hamiltonian(
        D=D_work,
        residual=residual_work,
        penalty=penalty,
    )

    vqc, schedule, num_params, theta_default = build_ry_chain_ansatz_with_biased_init(
        D=D_work,
        residual=residual_work,
        layers=layers,
        beta=1.6,
        p_zero_floor=0.02,
        p_bias_max=0.22,
        later_layer_noise=0.02,
        seed=1234,
    )

    if theta_init is None:
        theta = theta_default.copy()
    else:
        theta = np.asarray(theta_init, dtype=float).reshape(-1)
        if len(theta) != num_params:
            raise ValueError(
                f"theta_init 长度应为 {num_params}，但收到 {len(theta)}"
            )

    if auto_calibrate:
        candidate_paulis = []
        candidate_paulis.extend(list(h_pauli.keys())[: min(8, len(h_pauli))])

        for q in range(min(2, num_qubits)):
            p = ["I"] * num_qubits
            p[q] = "X"
            candidate_paulis.append("".join(p))

            p = ["I"] * num_qubits
            p[q] = "Y"
            candidate_paulis.append("".join(p))

            p = ["I"] * num_qubits
            p[q] = "Z"
            candidate_paulis.append("".join(p))

        if num_qubits >= 2:
            p = ["I"] * num_qubits
            p[0] = "Z"
            p[1] = "Z"
            candidate_paulis.append("".join(p))

        conv = calibrate_measurement_convention(
            vqc=vqc,
            params=theta,
            candidate_paulis=candidate_paulis,
            shots=8000,
        )
        bit_reversed = conv["bit_reversed"]
        y_rot_sign = conv["y_rot_sign"]
    else:
        bit_reversed = True
        y_rot_sign = 1

    history = []

    qvm = _fresh_qvm()
    try:
        for k in range(steps):
            t0 = time.perf_counter()
            E_before = measured_energy(
                vqc=vqc,
                params=theta,
                h_pauli=h_pauli,
                shots=shots_E,
                bit_reversed=bit_reversed,
                y_rot_sign=y_rot_sign,
                qvm=qvm,
            )
            t1 = time.perf_counter()

            A, C = build_A_C_by_measurement(
                vqc=vqc,
                params=theta,
                schedule=schedule,
                h_pauli=h_pauli,
                shots=shots_A_C,
                reg_floor=reg_floor,
                bit_reversed=bit_reversed,
                y_rot_sign=y_rot_sign,
                qvm=qvm,
            )
            t2 = time.perf_counter()

            A = np.asarray(A, dtype=np.float64)
            C = np.asarray(C, dtype=np.float64)

            A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
            C = np.nan_to_num(C, nan=0.0, posinf=1e6, neginf=-1e6)

            A = 0.5 * (A + A.T)

            ridge = max(reg_floor, 1e-1)
            A_reg = A + ridge * np.eye(A.shape[0])

            try:
                dot_theta = np.linalg.solve(A_reg, C)
            except np.linalg.LinAlgError:
                dot_theta = np.linalg.pinv(A_reg, rcond=1e-4) @ C

            dot_theta = np.nan_to_num(dot_theta, nan=0.0, posinf=1e3, neginf=-1e3)

            delta_theta = dtau * dot_theta
            step_norm = float(np.linalg.norm(delta_theta))
            if step_norm > max_delta_norm:
                delta_theta = delta_theta * (max_delta_norm / step_norm)

            theta = theta + delta_theta

            if not np.isfinite(theta).all():
                raise FloatingPointError("VarQITE 参数 theta 出现了非有限值。")

            t3 = time.perf_counter()

            E_after = measured_energy(
                vqc=vqc,
                params=theta,
                h_pauli=h_pauli,
                shots=shots_E,
                bit_reversed=bit_reversed,
                y_rot_sign=y_rot_sign,
                qvm=qvm,
            )
            t4 = time.perf_counter()

            history.append({
                "step": k,
                "E_before": float(E_before),
                "E_after": float(E_after),
                "dot_theta_norm": float(np.linalg.norm(dot_theta)),
                "theta": theta.copy(),
                "t_energy_before": float(t1 - t0),
                "t_build_A_C": float(t2 - t1),
                "t_update": float(t3 - t2),
                "t_energy_after": float(t4 - t3),
                "t_step_total": float(t4 - t0),
            })

            if log_steps and ((k % verbose_every == 0) or (k == steps - 1)):
                print(
                    f"[step {k:3d}] "
                    f"E_before = {E_before:.8f}, "
                    f"E_after = {E_after:.8f}, "
                    f"||dot_theta|| = {np.linalg.norm(dot_theta):.4e}"
                )
                print(
                    f"           timing: "
                    f"E_before={t1 - t0:.3f}s, "
                    f"A/C={t2 - t1:.3f}s, "
                    f"update={t3 - t2:.3f}s, "
                    f"E_after={t4 - t3:.3f}s, "
                    f"total={t4 - t0:.3f}s"
                )

        raw_counts = run_vqc_and_get_counts(
            vqc=vqc,
            params=theta,
            num_qubits=num_qubits,
            shots=shots_decode,
            basis_rotations=None,
            qvm=qvm,
        )

    finally:
        _finalize_qvm(qvm)

    delta_qite, decoded_count_map, best_energy = pick_best_delta_from_counts(
        counts=raw_counts,
        D=D_orig,
        residual=residual_orig,
        bit_reversed=bit_reversed,
    )

    return VarQITEResult(
        delta_qite=np.asarray(delta_qite, dtype=int),
        history=history,
        theta_final=np.asarray(theta, dtype=float),
        decoded_count_map=decoded_count_map,
        raw_counts=raw_counts,
        best_energy=float(best_energy),
        num_qubits=num_qubits,
        num_params=num_params,
        h_pauli=h_pauli,
    )