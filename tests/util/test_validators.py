import numpy as np
import pytest

from komm._util.validators import validate_index


@pytest.mark.parametrize(
    "value",
    [3, np.int64(3), np.uint8(3), np.array(3)],
)
def test_validate_index(value):
    integer = validate_index(value)
    assert integer == 3
    assert type(integer) is int


@pytest.mark.parametrize(
    "value",
    [3.0, np.float64(3.0), "3", None, np.array([3])],
)
def test_validate_index_not_integer(value):
    with pytest.raises(TypeError):
        validate_index(value)


def test_validate_index_low():
    assert validate_index(0) == 0
    assert validate_index(1, low=1) == 1
    with pytest.raises(ValueError, match="value must be at least 0"):
        validate_index(-1)
    with pytest.raises(ValueError, match="value must be at least 1"):
        validate_index(0, low=1)


def test_validate_index_high():
    assert validate_index(0, high=2) == 0
    assert validate_index(1, high=2) == 1
    for value in [-1, 2]:
        with pytest.raises(ValueError, match=r"value must be in \[0:2\)"):
            validate_index(value, high=2)
