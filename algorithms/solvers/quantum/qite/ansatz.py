from __future__ import annotations

"""
ansatz.py
=========

这里保留当前项目里对 measurement-based VarQITE 最友好的 ansatz：

    build_ry_chain_ansatz

即：
    每层所有 qubit 做一轮 RY
    层间接固定 CNOT chain

同时新增：
    make_biased_theta_init
    build_ry_chain_ansatz_with_biased_init

核心思想
--------
当前局部目标函数是

    C(delta) = || D delta - residual ||^2
             = delta^T G delta - 2 h^T delta + const

其中
    G = D^T D
    h = D^T residual

h_i 的符号和大小给出了 delta_i 更偏向取 +1 / -1 / 0 的方向。
因此我们不改 ansatz 结构，只改 theta_init：

- 若 h_i > 0，则更偏向 delta_i = +1
- 若 h_i < 0，则更偏向 delta_i = -1
- 若 h_i ≈ 0，则更偏向 delta_i = 0

这样通常能在不增加 steps / shots 的前提下，
让低能修正更容易在最终采样中出现。
"""

from typing import List, Tuple

import numpy as np

from ..common.backend import _HAS_PYQPANDA3, VQCircuit, RY, CNOT


# =========================================================
# 0) 小工具
# =========================================================
def _pair_qubits(var_idx: int) -> Tuple[int, int]:
    """
    第 var_idx 个 ternary 变量对应两个 qubit：
        m_i = 2*i
        p_i = 2*i + 1
    """
    return 2 * var_idx, 2 * var_idx + 1


def _compute_local_linear_signal(D: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """
    计算 h = D^T residual。
    """
    D = np.asarray(D, dtype=float)
    residual = np.asarray(residual, dtype=float).reshape(-1)
    return D.T @ residual


def _safe_tanh_scale(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    把实数向量压到 [-1, 1]，避免极端数值。
    """
    x = np.asarray(x, dtype=float)
    scale = max(1e-12, float(np.max(np.abs(x))))
    return np.tanh(beta * x / scale)


def _prob_to_ry_angle(p1: float) -> float:
    """
    单 qubit 经 RY(theta) 后测到 |1> 的概率是 sin^2(theta/2)。
    给定目标概率 p1，返回对应 theta。
    """
    p1 = float(np.clip(p1, 0.0, 1.0))
    return 2.0 * np.arcsin(np.sqrt(p1))


# =========================================================
# 1) 原始通用 ansatz：保留
# =========================================================
def build_ry_chain_ansatz(num_qubits: int, layers: int = 2):
    """
    原始通用 ansatz：
        每层所有 qubit 做 RY
        每层之间接一条 CNOT chain

    返回：
        vqc
        schedule: list[tuple]
            ("RY", q, param_idx)
            ("CNOT", c, t)
        num_params
    """
    if not _HAS_PYQPANDA3:
        raise ImportError("pyqpanda3 未安装，无法构造真实量子线路版 VarQITE。")

    vqc = VQCircuit()
    num_params = layers * num_qubits
    vqc.set_Param([num_params])

    schedule: List[tuple] = []
    param_idx = 0

    for layer in range(layers):
        # 参数层：所有 qubit 上做 RY
        for q in range(num_qubits):
            p = vqc.Param([param_idx], f"theta{param_idx}")
            vqc << RY(q, p)
            schedule.append(("RY", q, param_idx))
            param_idx += 1

        # 非最后一层后面接 chain entangler
        if layer != layers - 1:
            for q in range(num_qubits - 1):
                vqc << CNOT(q, q + 1)
                schedule.append(("CNOT", q, q + 1))

    return vqc, schedule, num_params


# =========================================================
# 2) 新增：结构化初始化
# =========================================================
def make_biased_theta_init(
    D: np.ndarray,
    residual: np.ndarray,
    layers: int = 2,
    beta: float = 1.6,
    p_zero_floor: float = 0.02,
    p_bias_max: float = 0.22,
    later_layer_noise: float = 0.02,
    seed: int = 1234,
) -> np.ndarray:
    """
    为原始 RY-chain ansatz 生成“问题结构化”的 theta_init。

    编码回顾
    --------
    当前项目中每个 ternary 变量 delta_i 对应两个 qubit：
        m_i = 2*i
        p_i = 2*i + 1

    2-bit ternary 解码约定：
        10 -> delta_i = -1
        00 -> delta_i =  0
        01 -> delta_i = +1
        11 -> invalid

    所以：
        若想让 delta_i 更偏向 +1，就应让 p_i 更容易被激发到 1
        若想让 delta_i 更偏向 -1，就应让 m_i 更容易被激发到 1

    参数说明
    --------
    beta:
        h 的压缩强度。越大，方向性偏置越明显。
    p_zero_floor:
        即使 h_i 很弱，也保留很小的激发概率，避免完全冻死在 |0...0>。
    p_bias_max:
        方向性偏置的最大额外激发概率。
    later_layer_noise:
        后续层不做强偏置，只加很小随机扰动，避免参数完全对称。
    """
    D = np.asarray(D, dtype=float)
    residual = np.asarray(residual, dtype=float).reshape(-1)

    nvar = D.shape[1]
    num_qubits = 2 * nvar
    num_params = layers * num_qubits

    h = _compute_local_linear_signal(D, residual)
    h_score = _safe_tanh_scale(h, beta=beta)   # in [-1, 1]

    theta = np.zeros(num_params, dtype=float)
    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # 第一层：按 h 给每个 (m_i, p_i) 一对 qubit 做方向性偏置
    # -----------------------------------------------------
    base = 0
    for i in range(nvar):
        m_i, p_i = _pair_qubits(i)
        score = float(h_score[i])

        # score > 0: 更偏向 delta_i = +1 -> p_i 更容易激发
        # score < 0: 更偏向 delta_i = -1 -> m_i 更容易激发
        p_plus = p_zero_floor + p_bias_max * max(score, 0.0)
        p_minus = p_zero_floor + p_bias_max * max(-score, 0.0)

        theta[base + m_i] = _prob_to_ry_angle(p_minus)
        theta[base + p_i] = _prob_to_ry_angle(p_plus)

    # -----------------------------------------------------
    # 后续层：只加小扰动，不再加强偏置
    # -----------------------------------------------------
    for layer in range(1, layers):
        base = layer * num_qubits
        theta[base: base + num_qubits] = later_layer_noise * rng.normal(size=num_qubits)

    return theta


# =========================================================
# 3) 一站式构造器：推荐 solver 调这个
# =========================================================
def build_ry_chain_ansatz_with_biased_init(
    D: np.ndarray,
    residual: np.ndarray,
    layers: int = 2,
    beta: float = 1.6,
    p_zero_floor: float = 0.02,
    p_bias_max: float = 0.22,
    later_layer_noise: float = 0.02,
    seed: int = 1234,
):
    """
    方便 solver 直接调用：
        返回 vqc, schedule, num_params, theta_init
    """
    nvar = int(np.asarray(D).shape[1])
    num_qubits = 2 * nvar

    vqc, schedule, num_params = build_ry_chain_ansatz(
        num_qubits=num_qubits,
        layers=layers,
    )

    theta_init = make_biased_theta_init(
        D=D,
        residual=residual,
        layers=layers,
        beta=beta,
        p_zero_floor=p_zero_floor,
        p_bias_max=p_bias_max,
        later_layer_noise=later_layer_noise,
        seed=seed,
    )

    return vqc, schedule, num_params, theta_init