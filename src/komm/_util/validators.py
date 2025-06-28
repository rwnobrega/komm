from typing import Literal

import numpy as np
import numpy.typing as npt

from ..types import Array1D, Array2D


def validate_log_base(value: float | str) -> float | Literal["e"]:
    if (isinstance(value, str) and value != "e") or (
        isinstance(value, float) and value <= 0.0
    ):
        raise ValueError("log base must be 'e' or a positive float")
    return value


def validate_probability(value: float) -> float:
    if not 0 <= value <= 1:
        raise ValueError("probability must be between 0 and 1")
    return value


def validate_pmf(value: npt.ArrayLike) -> Array1D[np.floating]:
    value = np.asarray(value, dtype=float)
    if not value.ndim == 1:
        raise ValueError("pmf must be a 1D array")
    if not np.all(value >= 0.0):
        raise ValueError("pmf must be non-negative")
    if not np.isclose(value.sum(), 1.0):
        raise ValueError("pmf must sum to 1.0")
    return value


def validate_transition_matrix(value: npt.ArrayLike) -> Array2D[np.floating]:
    value = np.asarray(value, dtype=float)
    if not value.ndim == 2:
        raise ValueError("transition matrix must be a 2D array")
    if not np.all(value >= 0.0):
        raise ValueError("transition matrix must be non-negative")
    if not np.allclose(value.sum(axis=1), 1.0):
        raise ValueError("rows of transition matrix must sum to 1.0")
    return value
