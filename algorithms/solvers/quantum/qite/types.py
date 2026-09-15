from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..common.pauli import PauliSum


@dataclass
class VarQITEResult:
    delta_qite: np.ndarray
    history: List[dict]
    theta_final: np.ndarray
    decoded_count_map: Dict[Tuple[int, ...], int]
    raw_counts: Dict[str, int]
    best_energy: float
    num_qubits: int
    num_params: int
    h_pauli: PauliSum