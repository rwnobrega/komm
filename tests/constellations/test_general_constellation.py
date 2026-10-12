from re import escape

import numpy as np
import pytest

import komm


def test_general_constellation():
    const = komm.Constellation([[1.0], [2.0], [3.0], [4.0]])
    np.testing.assert_equal(const.matrix, [[1.0], [2.0], [3.0], [4.0]])
    np.testing.assert_equal(const.order, 4)
    np.testing.assert_equal(const.dimension, 1)
    np.testing.assert_equal(const.mean(), 2.5)
    np.testing.assert_equal(const.mean_energy(), 7.5)
    np.testing.assert_equal(const.minimum_distance(), 1.0)
    np.testing.assert_equal(const.indices_to_symbols(0), 1.0)
    np.testing.assert_equal(const.indices_to_symbols(1), 2.0)
    np.testing.assert_equal(const.indices_to_symbols(2), 3.0)
    np.testing.assert_equal(const.indices_to_symbols(3), 4.0)
    np.testing.assert_equal(const.indices_to_symbols([0, 1]), [1.0, 2.0])
    np.testing.assert_equal(const.closest_indices([1.1, 1.9]), [0, 1])
    np.testing.assert_equal(const.closest_symbols([1.1, 1.9]), [1.0, 2.0])


def test_general_constellation_1d():
    const = komm.Constellation([1, 1j, -1, -1j])
    np.testing.assert_equal(const.matrix, [[1], [1j], [-1], [-1j]])
    assert const.dimension == 1


@pytest.mark.parametrize(
    "matrix, message",
    [
        ([[[1.0]]], "'matrix' must be a 2D-array"),
        ([], "'matrix' must be non-empty"),
        ([[]], "'matrix' must be non-empty"),
        ([[1, 2], [1, 2]], "rows of 'matrix' must be distinct"),
        ([[1, 2], [3, 4], [1, 2]], "rows of 'matrix' must be distinct"),
        ([1, 1j, 1], "rows of 'matrix' must be distinct"),
        ([[0.0], [-0.0]], "rows of 'matrix' must be distinct"),
    ],
)
def test_general_constellation_invalid(matrix, message):
    with pytest.raises(ValueError, match=escape(message)):
        komm.Constellation(matrix)
