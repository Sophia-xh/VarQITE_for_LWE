# recover_secret.py
from __future__ import annotations

import numpy as np

from algorithms.common.utils import solve_mod_linear_system


def recover_secret_from_lattice_point(A: np.ndarray, w: np.ndarray, q: int) -> np.ndarray:
    """
    由格点 w 恢复 secret:
        A s = w (mod q)
    """
    rhs = np.array(w, dtype=int) % q
    s_hat = solve_mod_linear_system(A, rhs, q)
    return s_hat