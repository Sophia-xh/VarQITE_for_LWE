from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from pyqpanda3.vqcircuit import VQCircuit
    from pyqpanda3.core import CPUQVM, QProg, RY, H, RX, CNOT, measure
    _HAS_PYQPANDA3 = True
except Exception:
    VQCircuit = None
    CPUQVM = None
    QProg = None
    RY = None
    H = None
    RX = None
    CNOT = None
    measure = None
    _HAS_PYQPANDA3 = False


def _require_pyqpanda3() -> None:
    if not _HAS_PYQPANDA3:
        raise ImportError("pyqpanda3 未安装，无法运行真实量子线路版 QITE/VarQITE。")


def _fresh_qvm():
    _require_pyqpanda3()
    qvm = CPUQVM()
    if hasattr(qvm, "init_qvm"):
        qvm.init_qvm()
    return qvm


def _finalize_qvm(qvm) -> None:
    if qvm is not None and hasattr(qvm, "finalize"):
        qvm.finalize()


def vqc_to_prog(vqc, params: np.ndarray):
    _require_pyqpanda3()
    cir = vqc(params.tolist()).at([0])
    prog = QProg()
    prog << cir
    return prog


def get_statevector(vqc, params: np.ndarray) -> np.ndarray:
    """
    正确读取 pyqpanda3 的态矢量。
    这里只用于 calibration，不在主循环里反复调用。
    """
    prog = vqc_to_prog(vqc, np.asarray(params, dtype=float))
    qvm = _fresh_qvm()
    try:
        if hasattr(qvm, "directly_run"):
            qvm.directly_run(prog)
        else:
            qvm.run(prog, 1)

        if hasattr(qvm, "get_qstate"):
            psi = np.array(qvm.get_qstate(), dtype=np.complex128)
        else:
            psi = np.array(qvm.result().get_state_vector(), dtype=np.complex128)

        psi = np.asarray(psi, dtype=np.complex128).reshape(-1)

        if not np.all(np.isfinite(psi)):
            raise RuntimeError("get_statevector() 得到的 psi 含有 NaN/Inf。")

        norm = np.linalg.norm(psi)
        if norm < 1e-12 or not np.isfinite(norm):
            raise RuntimeError(f"get_statevector() 得到的 psi 范数异常: norm={norm}")

        return psi / norm
    finally:
        _finalize_qvm(qvm)


def run_prog_and_get_counts(prog, shots: int, qvm=None) -> Dict[str, int]:
    """
    若 qvm=None，则内部创建并销毁；
    若传入 qvm，则复用同一个 QVM，不在这里 finalize。
    """
    own_qvm = qvm is None
    if own_qvm:
        qvm = _fresh_qvm()

    try:
        qvm.run(prog, shots)
        raw_counts = qvm.result().get_counts()
        out = {}
        for k, v in raw_counts.items():
            key = str(k).replace(" ", "")
            out[key] = int(v)
        return out
    finally:
        if own_qvm:
            _finalize_qvm(qvm)


def run_vqc_and_get_counts(
    vqc,
    params: np.ndarray,
    num_qubits: int,
    shots: int,
    basis_rotations: List[Tuple[int, str]] | None = None,
    qvm=None,
) -> Dict[str, int]:
    """
    对参数化线路执行：
        1) 构造基础电路
        2) 追加基旋转
        3) measure 全部 qubits
        4) 返回 counts

    qvm 可选传入，用于整段 VarQITE 复用同一个后端。
    """
    _require_pyqpanda3()
    prog = vqc_to_prog(vqc, np.asarray(params, dtype=float))

    if basis_rotations is not None:
        for q, op in basis_rotations:
            if op == "H":
                prog << H(q)
            elif op == "RX+PI/2":
                prog << RX(q, np.pi / 2.0)
            elif op == "RX-PI/2":
                prog << RX(q, -np.pi / 2.0)
            else:
                raise ValueError(f"未知基变换操作: {op}")

    for q in range(num_qubits):
        prog << measure(q, q)

    counts = run_prog_and_get_counts(prog, shots=shots, qvm=qvm)

    out = {}
    for k, v in counts.items():
        out[str(k).zfill(num_qubits)] = int(v)
    return out