import numpy as np
import pytest

from komm._lossless_coding.util import is_prefix_free, lexicographical_code


@pytest.mark.parametrize(
    "lengths, expected",
    [
        ([3], [(0, 0, 0)]),
        ([0, 1, 0], [(), (0,), ()]),
        ([1, 2, 3, 3], [(0,), (1, 0), (1, 1, 0), (1, 1, 1)]),
        ([0, 2, 2], [(), (0, 0), (0, 1)]),
        ([3, 1, 3, 2], [(1, 1, 0), (0,), (1, 1, 1), (1, 0)]),
        ([3, 3, 3, 3], [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]),
        ([2, 2, 5, 5], [(0, 0), (0, 1), (1, 0, 0, 0, 0), (1, 0, 0, 0, 1)]),
        ([2, 3, 3, 3, 3], [(0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)]),
    ],
)
def test_lexicographical_code_basic(lengths, expected):
    got = lexicographical_code(lengths)
    assert got == expected
    assert is_prefix_free(got)


def test_lexicographical_code_invalid():
    with pytest.raises(ValueError, match="'lengths' must be a 1D-array"):
        lexicographical_code(np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError, match="'lengths' must be non-negative"):
        lexicographical_code([-1, 0, 1])
