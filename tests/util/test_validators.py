from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from linecache import cache
from re import escape

import numpy as np
import pytest

from komm._util.validators import (
    validate_bit_order,
    validate_bool,
    validate_choice,
    validate_float,
    validate_integer,
    validate_integer_array,
    validate_log_base,
    validate_nonnegative_integer,
    validate_pmf,
    validate_positive_integer,
    validate_transition_matrix,
)


@pytest.mark.parametrize(
    "value",
    [3, np.int64(3), np.uint8(3), np.array(3)],
)
def test_validate_integer(value):
    integer = validate_integer(value, "x")
    assert integer == 3
    assert type(integer) is int


@pytest.mark.parametrize(
    "value, got",
    [
        (3.0, "float"),
        (np.float64(3.0), "float64"),
        ("3", "str"),
        (None, "NoneType"),
        (np.array([3]), "ndarray"),
    ],
)
def test_validate_integer_not_integer(value, got):
    with pytest.raises(TypeError, match=rf"'x' must be an integer \(got {got}\)"):
        validate_integer(value, "x")


def test_validate_integer_no_bounds():
    assert validate_integer(-5, "x") == -5


def test_validate_integer_low():
    assert validate_integer(0, "x", low=0) == 0
    assert validate_integer(1, "x", low=1) == 1
    with pytest.raises(ValueError, match=r"'x' must be at least 0 \(got -1\)"):
        validate_integer(-1, "x", low=0)
    with pytest.raises(ValueError, match=r"'x' must be at least 1 \(got 0\)"):
        validate_integer(0, "x", low=1)


def test_validate_integer_high():
    assert validate_integer(-1, "x", high=2) == -1
    with pytest.raises(ValueError, match=r"'x' must be less than 2 \(got 2\)"):
        validate_integer(2, "x", high=2)


def test_validate_integer_low_high():
    assert validate_integer(0, "x", low=0, high=2) == 0
    assert validate_integer(1, "x", low=0, high=2) == 1
    for value in [-1, 2]:
        with pytest.raises(ValueError, match=r"'x' must be in \[0:2\)"):
            validate_integer(value, "x", low=0, high=2)


def test_validate_integer_rule():
    assert validate_integer(3, "d", low=1, high=4, rule="1 <= d <= n") == 3
    for value in [0, 4]:
        message = f"'d' must satisfy 1 <= d <= n (got {value})"
        with pytest.raises(ValueError, match=escape(message)):
            validate_integer(value, "d", low=1, high=4, rule="1 <= d <= n")


def test_validate_positive_integer():
    integer = validate_positive_integer(np.int64(1), "n")
    assert integer == 1
    assert type(integer) is int
    for value in [0, -1]:
        message = f"'n' must be a positive integer (got {value})"
        with pytest.raises(ValueError, match=escape(message)):
            validate_positive_integer(value, "n")
    with pytest.raises(TypeError, match=r"'n' must be an integer \(got float\)"):
        validate_positive_integer(1.0, "n")  # type: ignore


def test_validate_nonnegative_integer():
    integer = validate_nonnegative_integer(np.int64(0), "n")
    assert integer == 0
    assert type(integer) is int
    message = "'n' must be a non-negative integer (got -1)"
    with pytest.raises(ValueError, match=escape(message)):
        validate_nonnegative_integer(-1, "n")
    with pytest.raises(TypeError, match=r"'n' must be an integer \(got float\)"):
        validate_nonnegative_integer(0.0, "n")  # type: ignore


@pytest.mark.parametrize("dtype", [np.int64, np.uint8])
def test_validate_integer_array(dtype):
    value = np.array([0, 1, 2], dtype=dtype)
    array = validate_integer_array(value, "x")
    assert array.dtype == dtype
    np.testing.assert_equal(array, [0, 1, 2])


def test_validate_integer_array_bool_and_empty():
    array = validate_integer_array([True, False], "x")
    assert np.issubdtype(array.dtype, np.integer)
    np.testing.assert_equal(array, [1, 0])
    array = validate_integer_array([], "x")
    assert np.issubdtype(array.dtype, np.integer)
    assert array.size == 0


@pytest.mark.parametrize(
    "value, got",
    [
        ([1.0, 2.0], "float64"),
        ([1.5, 2.0], "float64"),
        (["1", "0"], "<U1"),
        ([1, 2**70], "object"),
    ],
)
def test_validate_integer_array_not_integer(value, got):
    with pytest.raises(
        TypeError, match=rf"'x' must contain only integers \(got {got}\)"
    ):
        validate_integer_array(value, "x")


def test_validate_integer_array_no_bounds():
    np.testing.assert_equal(validate_integer_array([-5, 0, 5], "x"), [-5, 0, 5])


def test_validate_integer_array_low():
    validate_integer_array([0, 1], "x", low=0)
    with pytest.raises(ValueError, match="elements of 'x' must be at least 0"):
        validate_integer_array([0, -1], "x", low=0)
    with pytest.raises(ValueError, match="elements of 'x' must be at least 1"):
        validate_integer_array([0, 1], "x", low=1)


def test_validate_integer_array_high():
    validate_integer_array([-1, 1], "x", high=2)
    with pytest.raises(ValueError, match="elements of 'x' must be less than 2"):
        validate_integer_array([0, 2], "x", high=2)


def test_validate_integer_array_low_high():
    validate_integer_array([[0, 1], [1, 0]], "x", low=0, high=2)
    for value in [[0, -1], [0, 2]]:
        with pytest.raises(ValueError, match=r"elements of 'x' must be in \[0:2\)"):
            validate_integer_array(value, "x", low=0, high=2)


def test_validate_integer_array_shape():
    validate_integer_array([1, 2], "x", shape=(2,))
    for value, got in [([1, 2, 3], "(3,)"), ([], "(0,)"), ([[1, 2]], "(1, 2)")]:
        message = f"'x' must have shape (2,) (got {got})"
        with pytest.raises(ValueError, match=escape(message)):
            validate_integer_array(value, "x", shape=(2,))


@pytest.mark.parametrize(
    "value",
    [1, True, np.int64(1), 1.0, np.float64(1.0), np.float32(1.0), Fraction(1)],
)
def test_validate_float(value):
    real = validate_float(value)
    assert real == 1.0
    assert type(real) is float


@pytest.mark.parametrize(
    "value, got",
    [
        ("1", "str"),
        (None, "NoneType"),
        (1j, "complex"),
        (np.complex128(1), "complex128"),
        (Decimal(1), "Decimal"),
        (np.array(1.0), "ndarray"),
    ],
)
def test_validate_float_not_real(value, got):
    with pytest.raises(TypeError, match=rf"'value' must be a real number \(got {got}"):
        validate_float(value)


def test_validate_float_no_bounds():
    assert validate_float(-5.0) == -5.0


def test_validate_float_low():
    assert validate_float(0.0, low=0) == 0.0
    for value in [-0.1, float("nan")]:
        with pytest.raises(ValueError, match="'value' must be at least 0"):
            validate_float(value, low=0)


def test_validate_float_high():
    assert validate_float(1.0, high=1) == 1.0
    value = 1.1
    with pytest.raises(ValueError, match=r"'value' must be at most 1 \(got 1.1\)"):
        validate_float(value, high=1)


def test_validate_float_low_high():
    assert validate_float(0.0, low=0, high=1) == 0.0
    assert validate_float(1.0, low=0, high=1) == 1.0
    for value in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match=r"'value' must be in \[0, 1\]"):
            validate_float(value, low=0, high=1)


@pytest.mark.parametrize("value", [False, True, np.False_, np.True_])
def test_validate_bool(value):
    boolean = validate_bool(value)
    assert boolean == value
    assert type(boolean) is bool


@pytest.mark.parametrize(
    "value, got",
    [
        (1, "int"),
        ("False", "str"),
        (None, "NoneType"),
        (np.array(True), "ndarray"),
    ],
)
def test_validate_bool_not_bool(value, got):
    with pytest.raises(TypeError, match=rf"'value' must be a boolean \(got {got}\)"):
        validate_bool(value)


@dataclass
class Flagged:
    flag: object

    def __post_init__(self):
        validate_bool(self.flag)


def test_arg_name():
    flag = 1
    with pytest.raises(TypeError, match="'flag' must be a boolean"):
        validate_bool(flag)
    with pytest.raises(TypeError, match="'flag' must be a boolean"):
        validate_bool(
            flag,
        )
    with pytest.raises(TypeError, match="'flag' must be a boolean"):
        Flagged(1)


def test_arg_name_no_source():
    with pytest.raises(TypeError, match="'value' must be a boolean"):
        eval("validate_bool(1)", {"validate_bool": validate_bool})


def test_arg_name_stale_source(monkeypatch: pytest.MonkeyPatch):
    code = compile("validate_bool(1)", "<stale>", "eval")
    monkeypatch.setitem(cache, "<stale>", (3, None, ["x)\n"], "<stale>"))
    with pytest.raises(TypeError, match="'value' must be a boolean"):
        eval(code, {"validate_bool": validate_bool})


def test_validate_log_base():
    assert validate_log_base("e") == "e"
    base = validate_log_base(2)
    assert base == 2.0
    assert type(base) is float


@pytest.mark.parametrize("value", [0, -1.0, 1, 1.0, float("nan"), "f", ""])
def test_validate_log_base_invalid(value):
    with pytest.raises(ValueError, match="'value' must be 'e' or a positive real"):
        validate_log_base(value)


@pytest.mark.parametrize("value", [None, 1j])
def test_validate_log_base_not_real(value):
    with pytest.raises(TypeError, match="'value' must be a real number"):
        validate_log_base(value)


def test_validate_pmf():
    pmf = validate_pmf([1, 0], "x")
    assert pmf.dtype == np.float64
    np.testing.assert_equal(pmf, [1.0, 0.0])


@pytest.mark.parametrize(
    "value, message",
    [
        ([[0.5], [0.5]], "'x' must be a 1D-array"),
        ([1.5, -0.5], "'x' must be non-negative"),
        ([0.5, 0.6], "'x' must sum to 1.0"),
    ],
)
def test_validate_pmf_invalid(value, message):
    with pytest.raises(ValueError, match=message):
        validate_pmf(value, "x")


def test_validate_transition_matrix():
    matrix = validate_transition_matrix([[1, 0], [0, 1]], "x")
    assert matrix.dtype == np.float64
    np.testing.assert_equal(matrix, [[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    "value, message",
    [
        ([0.5, 0.5], "'x' must be a 2D-array"),
        ([[1.5, -0.5]], "'x' must be non-negative"),
        ([[0.5, 0.6]], "rows of 'x' must sum to 1.0"),
        ([[0.5, 0.5]], r"'x' must be square \(got shape \(1, 2\)\)"),
    ],
)
def test_validate_transition_matrix_invalid(value, message):
    with pytest.raises(ValueError, match=message):
        validate_transition_matrix(value, "x", square=True)


def test_validate_choice():
    x = "b"
    assert validate_choice(x, ["a", "b"]) == "b"
    x = "c"
    with pytest.raises(ValueError, match="'x' must be 'a' or 'b'"):
        validate_choice(x, ["a", "b"])
    x = 4
    with pytest.raises(ValueError, match="'x' must be 1, 2 or 3"):
        validate_choice(x, [1, 2, 3])


def test_arg_name_partial():
    bit_order = "LSB"
    with pytest.raises(ValueError, match="'bit_order' must be 'LSB-first' or"):
        validate_bit_order(bit_order)  # type: ignore
