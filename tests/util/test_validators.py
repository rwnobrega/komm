from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from komm._util.validators import (
    validate_float,
    validate_integer,
    validate_integer_array,
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
    with pytest.raises(ValueError, match="'x' must be at least 0"):
        validate_integer(-1, "x", low=0)
    with pytest.raises(ValueError, match="'x' must be at least 1"):
        validate_integer(0, "x", low=1)


def test_validate_integer_high():
    assert validate_integer(-1, "x", high=2) == -1
    with pytest.raises(ValueError, match="'x' must be less than 2"):
        validate_integer(2, "x", high=2)


def test_validate_integer_low_high():
    assert validate_integer(0, "x", low=0, high=2) == 0
    assert validate_integer(1, "x", low=0, high=2) == 1
    for value in [-1, 2]:
        with pytest.raises(ValueError, match=r"'x' must be in \[0:2\)"):
            validate_integer(value, "x", low=0, high=2)


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


@pytest.mark.parametrize(
    "value",
    [1, True, np.int64(1), 1.0, np.float64(1.0), np.float32(1.0), Fraction(1)],
)
def test_validate_float(value):
    real = validate_float(value, "x")
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
    with pytest.raises(TypeError, match=rf"'x' must be a real number \(got {got}\)"):
        validate_float(value, "x")


def test_validate_float_no_bounds():
    assert validate_float(-5.0, "x") == -5.0


def test_validate_float_low():
    assert validate_float(0.0, "x", low=0) == 0.0
    for value in [-0.1, float("nan")]:
        with pytest.raises(ValueError, match="'x' must be at least 0"):
            validate_float(value, "x", low=0)


def test_validate_float_high():
    assert validate_float(1.0, "x", high=1) == 1.0
    with pytest.raises(ValueError, match="'x' must be at most 1"):
        validate_float(1.1, "x", high=1)


def test_validate_float_low_high():
    assert validate_float(0.0, "x", low=0, high=1) == 0.0
    assert validate_float(1.0, "x", low=0, high=1) == 1.0
    for value in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match=r"'x' must be in \[0, 1\]"):
            validate_float(value, "x", low=0, high=1)
