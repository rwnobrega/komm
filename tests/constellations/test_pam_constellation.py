import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "delta": 2.0},
            {
                "matrix": [-1, 1],
                "mean_energy": 1.0,
            },
        ),
        (
            {"order": 4, "delta": 2.0},
            {
                "matrix": [-3, -1, 1, 3],
                "mean_energy": 5.0,
            },
        ),
        (
            {"order": 8, "delta": 2.0},
            {
                "matrix": [-7, -5, -3, -1, 1, 3, 5, 7],
                "mean_energy": 21.0,
            },
        ),
        (
            {"order": 8, "delta": 4.0},
            {
                "matrix": [-14, -10, -6, -2, 2, 6, 10, 14],
                "mean_energy": 84.0,
            },
        ),
        (
            {"order": 8, "delta": 1.0},
            {
                "matrix": [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
                "mean_energy": 5.25,
            },
        ),
    ],
)
def test_pam_parameters(params, expected):
    pam = komm.PAMConstellation(**params)
    np.testing.assert_allclose(pam.matrix.ravel(), expected["matrix"])
    assert pam.order == params["order"]
    assert pam.dimension == 1
    np.testing.assert_allclose(pam.mean(), 0.0)
    np.testing.assert_allclose(pam.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(pam.minimum_distance(), params["delta"])


@pytest.mark.parametrize(
    "order, indices, symbols",
    [
        (2, [0, 0, 1, 1, 0, 0], [-1, -1, +1, +1, -1, -1]),
        (4, [0, 2, 0, 3, 3, 1], [-3, +1, -3, +3, +3, -1]),
        (8, [1, 5, 7, 1, 2, 4], [-5, +3, +7, -5, -3, +1]),
    ],
)
def test_pam_indices_to_symbols(order, indices, symbols):
    pam = komm.PAMConstellation(order)
    np.testing.assert_equal(pam.indices_to_symbols(indices), symbols)


@pytest.mark.parametrize(
    "order, indices",
    [
        (2, [0, 0, 0, 1, 1, 1]),
        (4, [0, 0, 1, 2, 3, 3]),
        (8, [0, 0, 3, 4, 7, 7]),
    ],
)
def test_pam_closest_indices(order, indices):
    pam = komm.PAMConstellation(order)
    received = [-20.0, -7.1, -0.5, 1.5, 6.2, 100.0]
    np.testing.assert_allclose(pam.closest_indices(received), indices)
