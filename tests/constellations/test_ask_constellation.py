import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "base_amplitude": 1, "phase_offset": 0},
            {
                "matrix": [0, 1],
                "mean": 0.5,
                "mean_energy": 0.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "base_amplitude": 1, "phase_offset": 0},
            {
                "matrix": [0, 1, 2, 3],
                "mean": 1.5,
                "mean_energy": 3.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 8, "base_amplitude": 1, "phase_offset": 0},
            {
                "matrix": [0, 1, 2, 3, 4, 5, 6, 7],
                "mean": 3.5,
                "mean_energy": 17.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "base_amplitude": 1, "phase_offset": 0},
            {
                "matrix": [0, 1, 2, 3],
                "mean": 1.5,
                "mean_energy": 3.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "base_amplitude": 2, "phase_offset": 0},
            {
                "matrix": [0, 2, 4, 6],
                "mean": 3,
                "mean_energy": 14,
                "minimum_distance": 2,
            },
        ),
        (
            {"order": 4, "base_amplitude": 1, "phase_offset": 1 / 8},
            {
                "matrix": np.array([0, 1 + 1j, 2 + 2j, 3 + 3j]) / np.sqrt(2),
                "mean": (1.5 + 1.5j) / np.sqrt(2),
                "mean_energy": 3.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "base_amplitude": 0.5, "phase_offset": 1 / 2},
            {
                "matrix": [0, -0.5, -1, -1.5],
                "mean": -0.75,
                "mean_energy": 0.875,
                "minimum_distance": 0.5,
            },
        ),
        (
            {"order": 4, "base_amplitude": 2.5, "phase_offset": 1 / 4},
            {
                "matrix": [0, 2.5j, 5j, 7.5j],
                "mean": 3.75j,
                "mean_energy": 21.875,
                "minimum_distance": 2.5,
            },
        ),
    ],
)
def test_ask_parameters(params, expected):
    ask = komm.ASKConstellation(**params)
    np.testing.assert_allclose(ask.matrix.ravel(), expected["matrix"])
    assert ask.order == params["order"]
    assert ask.dimension == 1
    np.testing.assert_allclose(ask.mean(), expected["mean"])
    np.testing.assert_allclose(ask.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(ask.minimum_distance(), expected["minimum_distance"])


@pytest.mark.parametrize(
    "order, indices",
    [
        (2, [0, 0, 0, 1, 1, 1]),
        (4, [0, 0, 0, 1, 2, 3]),
        (8, [0, 0, 0, 1, 2, 7]),
    ],
)
def test_ask_closest_indices(order, indices):
    ask = komm.ASKConstellation(order)
    received = [-0.5, 0.25, 0.4, 0.65, 2.1, 10]
    assert np.allclose(ask.closest_indices(received), indices)
