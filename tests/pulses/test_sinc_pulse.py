import komm
from komm._pulses.AbstractPulse import AbstractPulse


def test_rectangular_pulse_protocol():
    pulse: AbstractPulse = komm.SincPulse()
    assert isinstance(pulse, AbstractPulse)
