import ast
import inspect
import linecache
from collections.abc import Sequence
from functools import partial
from itertools import islice
from numbers import Real
from operator import index
from typing import Literal, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from ..types import Array1D, Array2D

T = TypeVar("T")


def arg_name() -> str:
    # First argument of the outer validator call.
    frame = inspect.currentframe()
    while frame is not None and frame.f_globals is globals():
        frame = frame.f_back
    if frame is None:
        return "value"
    positions = frame.f_code.co_positions()
    lineno, end_lineno, col, end_col = next(islice(positions, frame.f_lasti // 2, None))
    if lineno is None or end_lineno is None or col is None or end_col is None:
        return "value"
    lines = linecache.getlines(frame.f_code.co_filename)[lineno - 1 : end_lineno]
    if not lines:
        return "value"
    # Columns are byte offsets.
    chunks = [line.encode() for line in lines]
    chunks[-1] = chunks[-1][:end_col]
    chunks[0] = chunks[0][col:]
    try:
        call = ast.parse(b"".join(chunks), mode="eval").body
    except SyntaxError:  # source changed since import
        return "value"
    if not isinstance(call, ast.Call) or not call.args:
        return "value"
    return ast.unparse(call.args[0]).removeprefix("self.")


def validate_log_base(value: float | str) -> float | Literal["e"]:
    if isinstance(value, str):
        if value == "e":
            return "e"
    else:
        value = validate_float(value)
        if value > 0 and value != 1:
            return value
    raise ValueError(f"'{arg_name()}' must be 'e' or a positive real other than 1")


def validate_pmf(
    value: npt.ArrayLike,
    *,
    size: int | None = None,
) -> Array1D[np.floating]:
    value = np.asarray(value)
    if not value.ndim == 1:
        raise ValueError(f"'{arg_name()}' must be a 1D-array")
    if size is not None and not value.size == size:
        raise ValueError(f"'{arg_name()}' must have size {size} (got {value.size})")
    if value.dtype.kind not in "biuf":  # bool, int, uint, float
        got = value.dtype
        raise TypeError(f"'{arg_name()}' must contain only real numbers (got {got})")
    value = value.astype(float, copy=False)
    if not np.all(value >= 0.0):
        raise ValueError(f"'{arg_name()}' must be non-negative")
    if not np.isclose(value.sum(), 1.0):
        raise ValueError(f"'{arg_name()}' must sum to 1.0")
    return value


def validate_transition_matrix(
    value: npt.ArrayLike,
    *,
    square: bool = False,
) -> Array2D[np.floating]:
    value = np.asarray(value)
    if not value.ndim == 2:
        raise ValueError(f"'{arg_name()}' must be a 2D-array")
    if value.dtype.kind not in "biuf":  # bool, int, uint, float
        got = value.dtype
        raise TypeError(f"'{arg_name()}' must contain only real numbers (got {got})")
    value = value.astype(float, copy=False)
    if not np.all(value >= 0.0):
        raise ValueError(f"'{arg_name()}' must be non-negative")
    if not np.allclose(value.sum(axis=1), 1.0):
        raise ValueError(f"rows of '{arg_name()}' must sum to 1.0")
    if square and value.shape[0] != value.shape[1]:
        raise ValueError(f"'{arg_name()}' must be square (got shape {value.shape})")
    return value


def validate_integer(
    value: SupportsIndex,
    *,
    low: int | None = None,
    high: int | None = None,
    rule: str | None = None,
) -> int:
    try:
        integer = index(value)
    except TypeError:
        got = type(value).__name__
        raise TypeError(f"'{arg_name()}' must be an integer (got {got})") from None
    if (low is None or integer >= low) and (high is None or integer < high):
        return integer
    if rule is not None:
        condition = f"satisfy {rule}"
    elif high is None:
        condition = f"be at least {low}"
    elif low is None:
        condition = f"be less than {high}"
    else:
        condition = f"be in [{low}:{high})"
    raise ValueError(f"'{arg_name()}' must {condition} (got {integer})")


def validate_positive_integer(value: SupportsIndex) -> int:
    integer = validate_integer(value)
    if not integer > 0:
        raise ValueError(f"'{arg_name()}' must be a positive integer (got {integer})")
    return integer


def validate_nonnegative_integer(value: SupportsIndex) -> int:
    integer = validate_integer(value)
    if not integer >= 0:
        raise ValueError(
            f"'{arg_name()}' must be a non-negative integer (got {integer})"
        )
    return integer


def validate_integer_array(
    value: npt.ArrayLike,
    *,
    low: int | None = None,
    high: int | None = None,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> npt.NDArray[np.integer]:
    value = np.asarray(value)
    if ndim is not None and not value.ndim == ndim:
        raise ValueError(
            f"'{arg_name()}' must be a {ndim}D-array (got shape {value.shape})"
        )
    if shape is not None and not value.shape == shape:
        raise ValueError(f"'{arg_name()}' must have shape {shape} (got {value.shape})")
    if value.size == 0:  # np.asarray([]) is float64
        return value.astype(int)
    if value.dtype == bool:
        value = value.astype(int)
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(
            f"'{arg_name()}' must contain only integers (got {value.dtype})"
        )
    if low is not None and high is not None:
        if not (value.min() >= low and value.max() < high):
            raise ValueError(f"elements of '{arg_name()}' must be in [{low}:{high})")
    elif low is not None and not value.min() >= low:
        raise ValueError(f"elements of '{arg_name()}' must be at least {low}")
    elif high is not None and not value.max() < high:
        raise ValueError(f"elements of '{arg_name()}' must be less than {high}")
    return value


def validate_binary_array(
    value: npt.ArrayLike,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> npt.NDArray[np.integer]:
    array = validate_integer_array(value, ndim=ndim, shape=shape)
    if array.size > 0 and not (array.min() >= 0 and array.max() <= 1):
        raise ValueError(f"elements of '{arg_name()}' must be 0 or 1")
    return array


def validate_float(
    value: float,
    *,
    low: float | None = None,
    high: float | None = None,
) -> float:
    if not isinstance(value, Real):
        got = type(value).__name__
        raise TypeError(f"'{arg_name()}' must be a real number (got {got})")
    if low is not None and high is not None:
        if not low <= value <= high:
            raise ValueError(f"'{arg_name()}' must be in [{low}, {high}] (got {value})")
    elif low is not None and not value >= low:
        raise ValueError(f"'{arg_name()}' must be at least {low} (got {value})")
    elif high is not None and not value <= high:
        raise ValueError(f"'{arg_name()}' must be at most {high} (got {value})")
    return float(value)


def validate_positive_float(value: float) -> float:
    real = validate_float(value)
    if not real > 0:
        raise ValueError(f"'{arg_name()}' must be a positive real number (got {real})")
    return real


def validate_nonnegative_float(value: float) -> float:
    real = validate_float(value)
    if not real >= 0:
        raise ValueError(
            f"'{arg_name()}' must be a non-negative real number (got {real})"
        )
    return real


def validate_float_array(
    value: npt.ArrayLike,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> npt.NDArray[np.floating]:
    value = np.asarray(value)
    if ndim is not None and not value.ndim == ndim:
        raise ValueError(
            f"'{arg_name()}' must be a {ndim}D-array (got shape {value.shape})"
        )
    if shape is not None and not value.shape == shape:
        raise ValueError(f"'{arg_name()}' must have shape {shape} (got {value.shape})")
    if value.dtype.kind not in "biuf":  # bool, int, uint, float
        got = value.dtype
        raise TypeError(f"'{arg_name()}' must contain only real numbers (got {got})")
    return value.astype(float, copy=False)


def validate_bool(value: object) -> bool:
    if type(value) not in (bool, np.bool_):
        got = type(value).__name__
        raise TypeError(f"'{arg_name()}' must be a boolean (got {got})")
    return bool(value)


def validate_choice(value: T, choices: Sequence[T]) -> T:
    if value not in choices:
        *init, last = map(repr, choices)
        raise ValueError(f"'{arg_name()}' must be {', '.join(init)} or {last}")
    return value


validate_bit_order = partial(validate_choice, choices=("LSB-first", "MSB-first"))

validate_decision_type = partial(validate_choice, choices=("hard", "soft"))
