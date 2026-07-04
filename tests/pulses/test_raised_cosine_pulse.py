import numpy as np
import pytest

import komm


def test_raised_cosine_pulse_repr():
    pulse = komm.RaisedCosinePulse(rolloff=0.25)
    assert repr(pulse) == "RaisedCosinePulse(rolloff=0.25)"
    pulse = komm.RaisedCosinePulse(rolloff=0.25).root()
    assert repr(pulse) == "RaisedCosinePulse(rolloff=0.25).root()"


def test_raised_cosine_pulse_autocorrelation_scalar_regression():
    α = 0.25
    pulse = komm.RaisedCosinePulse(rolloff=α)
    assert np.isclose(pulse.autocorrelation(1 / α), pulse.autocorrelation([1 / α])[0])
    pulse = komm.RaisedCosinePulse(rolloff=α).root()
    assert np.isclose(pulse.waveform(0.0), 1 + α * (4 / np.pi - 1))


@pytest.mark.parametrize("rolloff", [-0.1, 1.5, 2.0])
def test_raised_cosine_pulse_invalid_rolloff(rolloff):
    with pytest.raises(ValueError, match="must satisfy 0 <= rolloff <= 1"):
        komm.RaisedCosinePulse(rolloff=rolloff)
    with pytest.raises(ValueError, match="must satisfy 0 <= rolloff <= 1"):
        komm.RaisedCosinePulse(rolloff=rolloff).root()
