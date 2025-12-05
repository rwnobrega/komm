from typing import Literal

import numpy as np
import numpy.typing as npt

from ..types import Array2D


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


def validate_pmf(
    value: npt.ArrayLike,
    joint: bool = False,
) -> npt.NDArray[np.floating]:
    value = np.asarray(value, dtype=float)
    if joint:
        if not (len(value.shape) > 0 and len(set(value.shape)) == 1):
            raise ValueError(
                f"pmf must have equal dimensions (got shape {value.shape})"
            )
    else:
        if not value.ndim == 1:
            raise ValueError("pmf must be a 1D-array")
    if not np.all(value >= 0.0):
        raise ValueError("pmf must be non-negative")
    if not np.isclose(value.sum(), 1.0):
        raise ValueError("pmf must sum to 1.0")
    return value


def validate_transition_matrix(
    value: npt.ArrayLike,
    square: bool = False,
) -> Array2D[np.floating]:
    value = np.asarray(value, dtype=float)
    if not value.ndim == 2:
        raise ValueError("transition matrix must be a 2D array")
    if not np.all(value >= 0.0):
        raise ValueError("transition matrix must be non-negative")
    if not np.allclose(value.sum(axis=1), 1.0):
        raise ValueError("rows of transition matrix must sum to 1.0")
    if square and value.shape[0] != value.shape[1]:
        raise ValueError(
            "transition matrix must be square (got shape "
            f"({value.shape[0]}, {value.shape[1]}))"
        )
    return value


def validate_integer_range(
    value: npt.ArrayLike,
    *,
    low: int = 0,
    high: int = 2,
) -> npt.NDArray[np.integer]:
    value = np.asarray(value, dtype=int)
    if not (np.all(value >= low) and np.all(value < high)):
        raise ValueError(f"input contains invalid entries (expected in [{low}:{high}))")
    return value
