import numpy as np
import pytest

import komm

# fmt: off
cross32 = np.array([
             -5 - 3j, -5 - 1j, -5 + 1j, -5 + 3j,
    -3 - 5j, -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j, -3 + 5j,
    -1 - 5j, -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j, -1 + 5j,
    +1 - 5j, +1 - 3j, +1 - 1j, +1 + 1j, +1 + 3j, +1 + 5j,
    +3 - 5j, +3 - 3j, +3 - 1j, +3 + 1j, +3 + 3j, +3 + 5j,
             +5 - 3j, +5 - 1j, +5 + 1j, +5 + 3j,
])
# fmt: on


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 32},
            {
                "matrix": cross32,
                "order": 32,
                "mean_energy": 20.0,
                "minimum_distance": 2.0,
            },
        ),
        (
            {"order": 32, "delta": 1.0},
            {
                "matrix": cross32 / 2,
                "order": 32,
                "mean_energy": 5.0,
                "minimum_distance": 1.0,
            },
        ),
        (
            {"order": 32, "phase_offset": 1 / 4},
            {
                "matrix": 1j * cross32,
                "order": 32,
                "mean_energy": 20.0,
                "minimum_distance": 2.0,
            },
        ),
        (
            {"order": 128},
            {
                "order": 128,
                "mean_energy": 82.0,
                "minimum_distance": 2.0,
            },
        ),
        (
            {"order": 512, "delta": 1.0},
            {
                "order": 512,
                "mean_energy": 82.5,
                "minimum_distance": 1.0,
            },
        ),
    ],
)
def test_cross_qam_parameters(params, expected):
    cross = komm.CrossQAMConstellation(**params)
    if "matrix" in expected:
        np.testing.assert_allclose(cross.matrix.ravel(), expected["matrix"], atol=1e-12)
    assert cross.order == expected["order"]
    assert cross.dimension == 1
    np.testing.assert_allclose(cross.mean(), 0.0)
    np.testing.assert_allclose(cross.mean_energy(), expected["mean_energy"])
    np.testing.assert_allclose(cross.minimum_distance(), expected["minimum_distance"])


@pytest.mark.parametrize("order", [32, 128, 512, 2048])
def test_cross_qam_matrix_order(order):
    cross = komm.CrossQAMConstellation(order)
    np.testing.assert_equal(cross.matrix.shape[0], order)


@pytest.mark.parametrize("order", [8, 16, 31, 64, 100, 256, 1024])
def test_cross_qam_invalid(order):
    with pytest.raises(ValueError, match=r"2\^k"):
        komm.CrossQAMConstellation(order)


@pytest.mark.parametrize(
    "received, indices",
    [
        ([-5.1 - 3.2j, -0.2 + 0.1j, 4.9 + 4.8j, 3.1 - 4.9j], [0, 13, 31, 22]),
    ],
)
def test_cross_qam_closest_indices(received, indices):
    cross = komm.CrossQAMConstellation(32)
    np.testing.assert_equal(cross.closest_indices(received), indices)
