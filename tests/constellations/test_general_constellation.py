import numpy as np

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
