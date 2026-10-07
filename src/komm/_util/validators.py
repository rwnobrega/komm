from collections.abc import Sequence
from functools import partial
from numbers import Real
from operator import index
from typing import Literal, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from ..types import Array2D

T = TypeVar("T")


def validate_log_base(value: float | str, name: str) -> float | Literal["e"]:
    if isinstance(value, str):
        if value == "e":
            return "e"
    else:
        value = validate_float(value, name)
        if value > 0 and value != 1:
            return value
    raise ValueError(f"'{name}' must be 'e' or a positive real other than 1")


def validate_pmf(value: npt.ArrayLike, name: str) -> npt.NDArray[np.floating]:
    value = np.asarray(value, dtype=float)
    if not value.ndim == 1:
        raise ValueError(f"'{name}' must be a 1D-array")
    if not np.all(value >= 0.0):
        raise ValueError(f"'{name}' must be non-negative")
    if not np.isclose(value.sum(), 1.0):
        raise ValueError(f"'{name}' must sum to 1.0")
    return value


def validate_transition_matrix(
    value: npt.ArrayLike,
    name: str,
    *,
    square: bool = False,
) -> Array2D[np.floating]:
    value = np.asarray(value, dtype=float)
    if not value.ndim == 2:
        raise ValueError(f"'{name}' must be a 2D-array")
    if not np.all(value >= 0.0):
        raise ValueError(f"'{name}' must be non-negative")
    if not np.allclose(value.sum(axis=1), 1.0):
        raise ValueError(f"rows of '{name}' must sum to 1.0")
    if square and value.shape[0] != value.shape[1]:
        raise ValueError(f"'{name}' must be square (got shape {value.shape})")
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
            raise ValueError(f"'{name}' must be in [{low}:{high}) (got {integer})")
    elif low is not None and not integer >= low:
        raise ValueError(f"'{name}' must be at least {low} (got {integer})")
    elif high is not None and not integer < high:
        raise ValueError(f"'{name}' must be less than {high} (got {integer})")
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
            raise ValueError(f"'{name}' must be in [{low}, {high}] (got {value})")
    elif low is not None and not value >= low:
        raise ValueError(f"'{name}' must be at least {low} (got {value})")
    elif high is not None and not value <= high:
        raise ValueError(f"'{name}' must be at most {high} (got {value})")
    return float(value)


def validate_bool(value: object, name: str) -> bool:
    if type(value) not in (bool, np.bool_):
        got = type(value).__name__
        raise TypeError(f"'{name}' must be a boolean (got {got})")
    return bool(value)


def validate_choice(value: T, name: str, choices: Sequence[T]) -> T:
    if value not in choices:
        *init, last = map(repr, choices)
        raise ValueError(f"'{name}' must be {', '.join(init)} or {last}")
    return value


validate_bit_order = partial(validate_choice, choices=("LSB-first", "MSB-first"))

validate_decision_type = partial(validate_choice, choices=("hard", "soft"))
