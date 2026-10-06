import numpy as np
import pytest

from komm._util.validators import (
    validate_integer,
    validate_integer_array,
    validate_probability,
)


def test_validate_probability():
    assert validate_probability(0.0, "p") == 0.0
    assert validate_probability(1.0, "p") == 1.0
    for value in [-0.1, 1.1]:
        with pytest.raises(ValueError, match="'p' must be between 0 and 1"):
            validate_probability(value, "p")


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
