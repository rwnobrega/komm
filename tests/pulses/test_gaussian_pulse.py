import komm
import komm.abc


def test_gaussian_pulse_protocol():
    pulse: komm.abc.Pulse = komm.GaussianPulse()
    assert isinstance(pulse, komm.abc.Pulse)
