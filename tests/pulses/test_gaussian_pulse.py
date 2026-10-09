from re import escape

import pytest

import komm


@pytest.mark.parametrize("bandwidth", [0.0, -1.0])
def test_gaussian_pulse_invalid_bandwidth(bandwidth):
    message = f"'half_power_bandwidth' must be a positive real number (got {bandwidth})"
    with pytest.raises(ValueError, match=escape(message)):
        komm.GaussianPulse(half_power_bandwidth=bandwidth)


def test_gaussian_pulse_bandwidth_not_real():
    with pytest.raises(TypeError, match="'half_power_bandwidth' must be a real number"):
        komm.GaussianPulse(half_power_bandwidth="1.0")  # type: ignore
    pulse = komm.GaussianPulse(half_power_bandwidth=1)
    assert type(pulse.half_power_bandwidth) is float
