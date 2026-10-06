import numpy as np
import pytest

from komm._util.validators import validate_integer


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


def test_validate_integer_low():
    assert validate_integer(0, "x") == 0
    assert validate_integer(1, "x", low=1) == 1
    with pytest.raises(ValueError, match="'x' must be at least 0"):
        validate_integer(-1, "x")
    with pytest.raises(ValueError, match="'x' must be at least 1"):
        validate_integer(0, "x", low=1)


def test_validate_integer_high():
    assert validate_integer(0, "x", high=2) == 0
    assert validate_integer(1, "x", high=2) == 1
    for value in [-1, 2]:
        with pytest.raises(ValueError, match=r"'x' must be in \[0:2\)"):
            validate_integer(value, "x", high=2)
