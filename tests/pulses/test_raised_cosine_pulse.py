import numpy as np

import komm


def test_rc_autocorrelation_scalar_regression():
    α = 0.25
    pulse = komm.RaisedCosinePulse(rolloff=α)
    value = pulse.autocorrelation(1 / α)
    expected = pulse.autocorrelation([1 / α])[0]
    assert np.isclose(value, expected)
