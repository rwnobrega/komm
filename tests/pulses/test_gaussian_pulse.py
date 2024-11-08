import komm
from komm._pulses.AbstractPulse import AbstractPulse


def test_gaussian_pulse_protocol():
    pulse: AbstractPulse = komm.GaussianPulse()
    assert isinstance(pulse, AbstractPulse)
