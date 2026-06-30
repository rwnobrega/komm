import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "amplitude": 1.0},
            {
                "matrix": [[1], [-1]],
                "dimension": 1,
                "mean_energy": 1.0,
                "minimum_distance": 2.0,
            },
        ),
        (
            {"order": 4, "amplitude": 1.0},
            {
                "matrix": [[1, 0], [0, 1], [-1, 0], [0, -1]],
                "dimension": 2,
                "mean_energy": 1.0,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 6, "amplitude": 1.0},
            {
                "matrix": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ],
                "dimension": 3,
                "mean_energy": 1.0,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 4, "amplitude": 2.0},
            {
                "matrix": [[2, 0], [0, 2], [-2, 0], [0, -2]],
                "dimension": 2,
                "mean_energy": 4.0,
                "minimum_distance": 2 * np.sqrt(2),
            },
        ),
    ],
)
def test_biorthogonal_parameters(params, expected):
    const = komm.BiorthogonalConstellation(**params)
    np.testing.assert_allclose(const.matrix, expected["matrix"])
    assert const.order == params["order"]
    assert const.dimension == expected["dimension"]
    np.testing.assert_allclose(const.mean(), np.zeros(expected["dimension"]))
    np.testing.assert_allclose(const.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(const.minimum_distance(), expected["minimum_distance"])


@pytest.mark.parametrize("order", [1, 3, 5, 7])
def test_biorthogonal_odd_order_raises(order):
    with pytest.raises(ValueError, match="even"):
        komm.BiorthogonalConstellation(order)


@pytest.mark.parametrize("order", [4, 6, 8])
def test_biorthogonal_antipodal_structure(order):
    const = komm.BiorthogonalConstellation(order)
    np.testing.assert_allclose(const.matrix[order // 2 :], -const.matrix[: order // 2])


@pytest.mark.parametrize(
    "order, indices, symbols",
    [
        (2, [0, 1], [1, -1]),
        (4, [0, 3], [1, 0, 0, -1]),
        (4, [[1, 2], [0, 3]], [[0, 1, -1, 0], [1, 0, 0, -1]]),
    ],
)
def test_biorthogonal_indices_to_symbols(order, indices, symbols):
    const = komm.BiorthogonalConstellation(order)
    np.testing.assert_equal(const.indices_to_symbols(indices), symbols)


def test_biorthogonal_closest_indices():
    const = komm.BiorthogonalConstellation(4)
    received = [0.9, 0.1, 0.1, 0.9, -0.8, 0.2, 0.1, -0.7]
    np.testing.assert_equal(const.closest_indices(received), [0, 1, 2, 3])


def test_biorthogonal_closest_indices_order_2():
    const = komm.BiorthogonalConstellation(2)
    received = [0.5, -0.3, 2.0, -1.0]
    np.testing.assert_equal(const.closest_indices(received), [0, 1, 0, 1])
