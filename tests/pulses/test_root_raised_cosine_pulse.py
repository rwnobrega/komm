import komm
from komm._pulses.AbstractPulse import AbstractPulse


def test_root_raised_cosine_pulse_protocol():
    pulse: AbstractPulse = komm.RootRaisedCosinePulse()
    assert isinstance(pulse, AbstractPulse)
