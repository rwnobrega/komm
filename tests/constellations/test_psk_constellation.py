import numpy as np
import pytest

import komm

w8 = (1 + 1j) / np.sqrt(2)
w8c = np.conj(w8)


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "amplitude": 1, "phase_offset": 0},
            {
                "matrix": [1, -1],
                "mean_energy": 1,
                "minimum_distance": 2,
            },
        ),
        (
            {"order": 2, "amplitude": 0.5, "phase_offset": 1 / 4},
            {
                "matrix": [0.5j, -0.5j],
                "mean_energy": 0.25,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "amplitude": 1, "phase_offset": 0},
            {
                "matrix": [1, 1j, -1, -1j],
                "mean_energy": 1,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 4, "amplitude": 2, "phase_offset": 1 / 4},
            {
                "matrix": [2j, -2, -2j, 2],
                "mean_energy": 4,
                "minimum_distance": 2 * np.sqrt(2),
            },
        ),
        (
            {"order": 4, "amplitude": 3, "phase_offset": 1 / 8},
            {
                "matrix": [3 * w8, -3 * w8c, -3 * w8, 3 * w8c],
                "mean_energy": 9,
                "minimum_distance": 3 * np.sqrt(2),
            },
        ),
        (
            {"order": 8, "amplitude": 1, "phase_offset": 0},
            {
                "matrix": [1, w8, 1j, -w8c, -1, -w8, -1j, w8c],
                "mean_energy": 1,
                "minimum_distance": np.sqrt(2 - np.sqrt(2)),
            },
        ),
    ],
)
def test_psk_parameters(params, expected):
    psk = komm.PSKConstellation(**params)
    np.testing.assert_allclose(psk.matrix.ravel(), expected["matrix"])
    assert psk.order == params["order"]
    assert psk.dimension == 1
    np.testing.assert_allclose(psk.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(psk.mean(), 0)
    np.testing.assert_allclose(psk.minimum_distance(), expected["minimum_distance"])
