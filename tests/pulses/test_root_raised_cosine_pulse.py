import komm
import komm.abc


def test_root_raised_cosine_pulse_protocol():
    pulse: komm.abc.Pulse = komm.RootRaisedCosinePulse()
    assert isinstance(pulse, komm.abc.Pulse)
