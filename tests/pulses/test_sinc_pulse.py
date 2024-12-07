import komm
import komm.abc


def test_rectangular_pulse_protocol():
    pulse: komm.abc.Pulse = komm.SincPulse()
    assert isinstance(pulse, komm.abc.Pulse)
