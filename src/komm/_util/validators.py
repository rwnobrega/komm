from operator import index
from typing import Literal, SupportsIndex

import numpy as np
import numpy.typing as npt

from ..types import Array2D


def validate_log_base(value: float | str) -> float | Literal["e"]:
    if isinstance(value, str):
        if value != "e":
            raise ValueError("log base must be 'e' or a positive real other than 1")
    elif value <= 0.0 or value == 1.0:
        raise ValueError("log base must be 'e' or a positive real other than 1")
    return value


def validate_probability(value: float) -> float:
    if not 0 <= value <= 1:
        raise ValueError("probability must be between 0 and 1")
    return value


def validate_pmf(value: npt.ArrayLike) -> npt.NDArray[np.floating]:
    value = np.asarray(value, dtype=float)
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


def validate_integer_array(value: npt.ArrayLike) -> npt.NDArray[np.integer]:
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.integer):
        return value
    if not np.all(np.isfinite(value) & (value == np.floor(value))):
        raise ValueError("input must contain only integers")
    return value.astype(int)


def validate_integer_range(
    value: npt.ArrayLike,
    *,
    low: int = 0,
    high: int,
) -> npt.NDArray[np.integer]:
    value = validate_integer_array(value)
    if not (np.all(value >= low) and np.all(value < high)):
        raise ValueError(f"input contains invalid entries (expected in [{low}:{high}))")
    return value


def validate_index(
    value: SupportsIndex,
    *,
    low: int = 0,
    high: int | None = None,
) -> int:
    value = index(value)
    if high is None and not value >= low:
        raise ValueError(f"value must be at least {low}")
    if high is not None and not low <= value < high:
        raise ValueError(f"value must be in [{low}:{high})")
    return value
