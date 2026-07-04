import pytest

import komm


@pytest.mark.parametrize("bandwidth", [0.0, -1.0])
def test_gaussian_pulse_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError, match="must be positive"):
        komm.GaussianPulse(half_power_bandwidth=bandwidth)
