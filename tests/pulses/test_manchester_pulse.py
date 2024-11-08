import komm
from komm._pulses.AbstractPulse import AbstractPulse


def test_manchester_pulse_protocol():
    pulse: AbstractPulse = komm.ManchesterPulse()
    assert isinstance(pulse, AbstractPulse)
