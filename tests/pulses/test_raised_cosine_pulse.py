import komm
import komm.abc


def test_raised_cosine_pulse_protocol():
    pulse: komm.abc.Pulse = komm.RaisedCosinePulse()
    assert isinstance(pulse, komm.abc.Pulse)
