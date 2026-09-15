# lwe_instance.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from algorithms.common.utils import centered_mod, rank_mod_prime


@dataclass
class LWEInstance:
    A: np.ndarray
    s_centered: np.ndarray
    s_mod_q: np.ndarray
    e: np.ndarray
    c_mod_q: np.ndarray
    c_centered: np.ndarray
    w_true: np.ndarray
    q: int


def generate_small_lwe_instance(
    n: int = 3,
    m: int | None = None,
    q: int = 17,
    seed: int = 0,
    secret_probs=(0.25, 0.5, 0.25), # w=1/2
    noise_probs=(0.125, 0.75, 0.125), # tau=1/4
) -> LWEInstance:
    """
    生成一个小规模 LWE 实例，默认:
    - secret 取 ternary {-1,0,1}
    - noise  取 ternary {-1,0,1}
    - 为了 demo 简单，默认 m=n 且 A 在模 q 下满秩
    """
    rng = np.random.default_rng(seed)
    if m is None:
        m = n

    # 生成满列秩 A（为了后面恢复 secret 简单）
    while True:
        A = rng.integers(0, q, size=(m, n), dtype=int)
        if rank_mod_prime(A, q) == n:
            break

    while True:
        s_centered = rng.choice([-1, 0, 1], size=n, p=secret_probs)
        if np.any(s_centered != 0):
            break
    s_mod_q = s_centered % q
    e = rng.choice([-1, 0, 1], size=m, p=noise_probs)

    c_mod_q = (A @ s_mod_q + e) % q
    c_centered = centered_mod(c_mod_q, q)

    # 对应中心提升后的“真实格点”
    # c_centered = w_true + e
    w_true = c_centered - e

    return LWEInstance(
        A=A,
        s_centered=s_centered,
        s_mod_q=s_mod_q,
        e=e,
        c_mod_q=c_mod_q,
        c_centered=c_centered,
        w_true=w_true,
        q=q,
    )