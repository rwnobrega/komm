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
            {"orders": (4, 1)},
            {
                "matrix": [-3, -1, 1, 3],
                "order": 4,
                "mean_energy": 5.0,
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
    with pytest.raises(ValueError, match=r"'orders' must be at least 2 \(got 1\)"):
        komm.QAMConstellation(1)
    with pytest.raises(TypeError, match="'orders' must be an integer"):
        komm.QAMConstellation(16.0)  # type: ignore
    with pytest.raises(ValueError, match=r"'orders' must not be \(1, 1\)"):
        komm.QAMConstellation((1, 1))
    with pytest.raises(ValueError, match="elements of 'orders' must be at least 1"):
        komm.QAMConstellation((4, 0))
    with pytest.raises(ValueError, match=r"'orders' must have shape \(2,\)"):
        komm.QAMConstellation((4,))  # type: ignore
    with pytest.raises(TypeError, match="'orders' must contain only integers"):
        komm.QAMConstellation((4, 2.0))  # type: ignore


@pytest.mark.parametrize(
    "deltas, message",
    [
        (0.0, r"'deltas' must be a positive real number \(got 0.0\)"),
        ((2.0, -1.0), "elements of 'deltas' must be positive"),
        ((2.0,), r"'deltas' must have shape \(2,\) \(got \(1,\)\)"),
    ],
)
def test_qam_invalid_deltas(deltas, message):
    with pytest.raises(ValueError, match=message):
        komm.QAMConstellation(16, deltas=deltas)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"deltas": 2j}, "'deltas' must be a real number"),
        ({"deltas": (2.0, "4")}, "'deltas' must contain only real numbers"),
        ({"phase_offset": "0"}, "'phase_offset' must be a real number"),
    ],
)
def test_qam_not_real(kwargs, message):
    with pytest.raises(TypeError, match=message):
        komm.QAMConstellation(16, **kwargs)
