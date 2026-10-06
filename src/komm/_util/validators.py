from numbers import Real
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


def validate_integer(
    value: SupportsIndex,
    name: str,
    *,
    low: int | None = None,
    high: int | None = None,
) -> int:
    try:
        integer = index(value)
    except TypeError:
        got = type(value).__name__
        raise TypeError(f"'{name}' must be an integer (got {got})") from None
    if low is not None and high is not None:
        if not low <= integer < high:
            raise ValueError(f"'{name}' must be in [{low}:{high})")
    elif low is not None and not integer >= low:
        raise ValueError(f"'{name}' must be at least {low}")
    elif high is not None and not integer < high:
        raise ValueError(f"'{name}' must be less than {high}")
    return integer


def validate_integer_array(
    value: npt.ArrayLike,
    name: str,
    *,
    low: int | None = None,
    high: int | None = None,
) -> npt.NDArray[np.integer]:
    value = np.asarray(value)
    if value.size == 0:  # np.asarray([]) is float64
        return value.astype(int)
    if value.dtype == bool:
        value = value.astype(int)
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"'{name}' must contain only integers (got {value.dtype})")
    if low is not None and high is not None:
        if not (value.min() >= low and value.max() < high):
            raise ValueError(f"elements of '{name}' must be in [{low}:{high})")
    elif low is not None and not value.min() >= low:
        raise ValueError(f"elements of '{name}' must be at least {low}")
    elif high is not None and not value.max() < high:
        raise ValueError(f"elements of '{name}' must be less than {high}")
    return value


def validate_float(
    value: float,
    name: str,
    *,
    low: float | None = None,
    high: float | None = None,
) -> float:
    if not isinstance(value, Real):
        got = type(value).__name__
        raise TypeError(f"'{name}' must be a real number (got {got})")
    if low is not None and high is not None:
        if not low <= value <= high:
            raise ValueError(f"'{name}' must be in [{low}, {high}]")
    elif low is not None and not value >= low:
        raise ValueError(f"'{name}' must be at least {low}")
    elif high is not None and not value <= high:
        raise ValueError(f"'{name}' must be at most {high}")
    return float(value)
