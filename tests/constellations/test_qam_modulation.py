import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"orders": 4},
            {
                # fmt: off
                "matrix": [
                    -1 - 1j, -1 + 1j,
                    +1 - 1j, +1 + 1j,
                ],
                # fmt: on
                "order": 4,
                "mean_energy": 2.0,
            },
        ),
        (
            {"orders": (4, 2)},
            {
                # fmt: off
                "matrix": [
                    -3 - 1j, -3 + 1j,
                    -1 - 1j, -1 + 1j,
                    +1 - 1j, +1 + 1j,
                    +3 - 1j, +3 + 1j,
                ],
                # fmt: on
                "order": 8,
                "mean_energy": 6.0,
            },
        ),
        (
            {"orders": 16},
            {
                # fmt: off
                "matrix": [
                    -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
                    -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                    +1 - 3j, +1 - 1j, +1 + 1j, +1 + 3j,
                    +3 - 3j, +3 - 1j, +3 + 1j, +3 + 3j,
                ],
                # fmt: on
                "order": 16,
                "mean_energy": 10.0,
            },
        ),
    ],
)
def test_qam_parameters(params, expected):
    pam = komm.QAMConstellation(**params)
    np.testing.assert_allclose(pam.matrix.ravel(), expected["matrix"])
    assert pam.order == expected["order"]
    assert pam.dimension == 1
    np.testing.assert_allclose(pam.mean(), 0.0)
    np.testing.assert_allclose(pam.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(pam.minimum_distance(), 2.0)


def test_qam_invalid():
    with pytest.raises(ValueError, match="must be a perfect square"):
        komm.QAMConstellation(8)
