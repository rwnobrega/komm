import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"orders": (4, 4), "amplitudes": (1.0, 2.0)},
            {
                "matrix": [1, 1j, -1, -1j, 2, 2j, -2, -2j],
                "order": 8,
                "mean_energy": 2.5,
                "minimum_distance": 1,
            },
        ),
        (
            {
                "orders": (8, 8),
                "amplitudes": (1.0, 2.0),
                "phase_offsets": (0.0, 1 / 16),
            },
            {
                "matrix": [
                    +1.000 + 0.000j,
                    +0.707 + 0.707j,
                    +0.000 + 1.000j,
                    -0.707 + 0.707j,
                    -1.000 + 0.000j,
                    -0.707 - 0.707j,
                    +0.000 - 1.000j,
                    +0.707 - 0.707j,
                    +1.848 + 0.765j,
                    +0.765 + 1.848j,
                    -0.765 + 1.848j,
                    -1.848 + 0.765j,
                    -1.848 - 0.765j,
                    -0.765 - 1.848j,
                    +0.765 - 1.848j,
                    +1.848 - 0.765j,
                ],
                "order": 16,
                "mean_energy": 2.5,
                "minimum_distance": np.sqrt(2 - np.sqrt(2)),
            },
        ),
        (
            {
                "orders": (4, 12),
                "amplitudes": (np.sqrt(2), 3.0),
                "phase_offsets": (1 / 8, 0.0),
            },
            {
                "matrix": [
                    +1.000 + 1.000j,
                    -1.000 + 1.000j,
                    -1.000 - 1.000j,
                    +1.000 - 1.000j,
                    +3.000 + 0.000j,
                    +2.598 + 1.500j,
                    +1.500 + 2.598j,
                    +0.000 + 3.000j,
                    -1.500 + 2.598j,
                    -2.598 + 1.500j,
                    -3.000 + 0.000j,
                    -2.598 - 1.500j,
                    -1.500 - 2.598j,
                    +0.000 - 3.000j,
                    +1.500 - 2.598j,
                    +2.598 - 1.500j,
                ],
                "order": 16,
                "mean_energy": 7.25,
                "minimum_distance": 3 * np.sqrt(2 - np.sqrt(3)),
            },
        ),
    ],
)
def test_apsk_parameters(params, expected):
    apsk = komm.APSKConstellation(**params)
    np.testing.assert_allclose(apsk.matrix.ravel(), expected["matrix"], atol=1e-3)
    assert apsk.order == expected["order"]
    assert apsk.dimension == 1
    np.testing.assert_allclose(apsk.mean(), 0.0)
    np.testing.assert_allclose(apsk.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(apsk.minimum_distance(), expected["minimum_distance"])
