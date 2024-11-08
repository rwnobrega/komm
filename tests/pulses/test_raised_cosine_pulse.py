import komm
from komm._pulses.AbstractPulse import AbstractPulse


def test_raised_cosine_pulse_protocol():
    pulse: AbstractPulse = komm.RaisedCosinePulse()
    assert isinstance(pulse, AbstractPulse)
