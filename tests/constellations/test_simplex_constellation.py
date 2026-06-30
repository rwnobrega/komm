import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "base_amplitude": 1.0},
            {
                "matrix": [[0.5, -0.5], [-0.5, 0.5]],
                "dimension": 2,
                "mean_energy": 0.5,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 3, "base_amplitude": 1.0},
            {
                "matrix": [
                    [2 / 3, -1 / 3, -1 / 3],
                    [-1 / 3, 2 / 3, -1 / 3],
                    [-1 / 3, -1 / 3, 2 / 3],
                ],
                "dimension": 3,
                "mean_energy": 2 / 3,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 4, "base_amplitude": 1.0},
            {
                "matrix": [
                    [0.75, -0.25, -0.25, -0.25],
                    [-0.25, 0.75, -0.25, -0.25],
                    [-0.25, -0.25, 0.75, -0.25],
                    [-0.25, -0.25, -0.25, 0.75],
                ],
                "dimension": 4,
                "mean_energy": 0.75,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 4, "base_amplitude": 2.0},
            {
                "matrix": [
                    [1.5, -0.5, -0.5, -0.5],
                    [-0.5, 1.5, -0.5, -0.5],
                    [-0.5, -0.5, 1.5, -0.5],
                    [-0.5, -0.5, -0.5, 1.5],
                ],
                "dimension": 4,
                "mean_energy": 3.0,
                "minimum_distance": 2 * np.sqrt(2),
            },
        ),
    ],
)
def test_simplex_parameters(params, expected):
    const = komm.SimplexConstellation(**params)
    np.testing.assert_allclose(const.matrix, expected["matrix"])
    assert const.order == params["order"]
    assert const.dimension == expected["dimension"]
    np.testing.assert_allclose(
        const.mean(), np.zeros(expected["dimension"]), atol=1e-12
    )
    np.testing.assert_allclose(const.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(const.minimum_distance(), expected["minimum_distance"])


@pytest.mark.parametrize("order", [2, 3, 4, 8])
def test_simplex_is_zero_mean_orthogonal(order):
    orth = komm.OrthogonalConstellation(order).matrix
    simp = komm.SimplexConstellation(order).matrix
    np.testing.assert_allclose(simp, orth - orth.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("order", [2, 3, 4, 8])
def test_simplex_equicorrelated(order):
    const = komm.SimplexConstellation(order)
    gram = const.matrix @ const.matrix.T
    energy = const.mean_energy()
    off_diagonal = gram[~np.eye(order, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, -energy / (order - 1), atol=1e-12)


@pytest.mark.parametrize(
    "order, indices, symbols",
    [
        (
            2,
            [0, 1],
            [0.5, -0.5, -0.5, 0.5],
        ),
        (
            4,
            [1, 3],
            [-0.25, 0.75, -0.25, -0.25, -0.25, -0.25, -0.25, 0.75],
        ),
    ],
)
def test_simplex_indices_to_symbols(order, indices, symbols):
    const = komm.SimplexConstellation(order)
    np.testing.assert_allclose(const.indices_to_symbols(indices), symbols)


def test_simplex_closest_indices():
    const = komm.SimplexConstellation(4)
    received = np.array([
        [0.9, -0.1, -0.2, -0.3],
        [-0.3, 0.8, -0.1, -0.2],
        [-0.2, -0.1, -0.3, 0.7],
    ])
    np.testing.assert_equal(const.closest_indices(received.ravel()), [0, 1, 3])
