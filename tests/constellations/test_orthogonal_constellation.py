import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "amplitude": 1.0},
            {
                "matrix": [[1, 0], [0, 1]],
                "dimension": 2,
                "mean": [0.5, 0.5],
                "mean_energy": 1.0,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 4, "amplitude": 1.0},
            {
                "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "dimension": 4,
                "mean": [0.25, 0.25, 0.25, 0.25],
                "mean_energy": 1.0,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 4, "amplitude": 2.0},
            {
                "matrix": [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]],
                "dimension": 4,
                "mean": [0.5, 0.5, 0.5, 0.5],
                "mean_energy": 4.0,
                "minimum_distance": 2 * np.sqrt(2),
            },
        ),
    ],
)
def test_orthogonal_parameters(params, expected):
    const = komm.OrthogonalConstellation(**params)
    np.testing.assert_allclose(const.matrix, expected["matrix"])
    assert const.order == params["order"]
    assert const.dimension == expected["dimension"]
    np.testing.assert_allclose(const.mean(), expected["mean"])
    np.testing.assert_allclose(const.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(const.minimum_distance(), expected["minimum_distance"])


@pytest.mark.parametrize("order", [2, 4, 8])
def test_orthogonal_symbols_are_orthonormal(order):
    const = komm.OrthogonalConstellation(order)
    gram = const.matrix @ const.matrix.T
    np.testing.assert_allclose(gram, np.eye(order))


@pytest.mark.parametrize(
    "order, indices, symbols",
    [
        (2, [0, 1], [1, 0, 0, 1]),
        (4, [1, 3], [0, 1, 0, 0, 0, 0, 0, 1]),
        (4, [[1, 3], [0, 2]], [[0, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1, 0]]),
    ],
)
def test_orthogonal_indices_to_symbols(order, indices, symbols):
    const = komm.OrthogonalConstellation(order)
    np.testing.assert_equal(const.indices_to_symbols(indices), symbols)


def test_orthogonal_closest_indices():
    const = komm.OrthogonalConstellation(4)
    received = [2.0, 1.0, 0.5, 0.1, -1.0, 3.0, 0.0, 0.2, 0.1, 0.1, 0.1, 5.0]
    np.testing.assert_equal(const.closest_indices(received), [0, 1, 3])


def test_orthogonal_closest_indices_shape():
    const = komm.OrthogonalConstellation(4)
    received = [[2.0, 1.0, 0.5, 0.1], [0.1, 0.1, 0.1, 5.0]]
    np.testing.assert_equal(const.closest_indices(received), [[0], [3]])
