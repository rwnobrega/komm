import komm
import komm.abc


def test_manchester_pulse_protocol():
    pulse: komm.abc.Pulse = komm.ManchesterPulse()
    assert isinstance(pulse, komm.abc.Pulse)
