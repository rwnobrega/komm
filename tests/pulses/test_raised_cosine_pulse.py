import numpy as np
import pytest

import komm


def test_rc_autocorrelation_scalar_regression():
    α = 0.25
    pulse = komm.RaisedCosinePulse(rolloff=α)
    value = pulse.autocorrelation(1 / α)
    expected = pulse.autocorrelation([1 / α])[0]
    assert np.isclose(value, expected)


@pytest.mark.parametrize("rolloff", [-0.1, 1.5, 2.0])
def test_rc_invalid_rolloff(rolloff):
    with pytest.raises(ValueError, match="must satisfy 0 <= rolloff <= 1"):
        komm.RaisedCosinePulse(rolloff=rolloff)
