import numpy as np
import pytest

import komm


def test_rrc_waveform_scalar_regression():
    α = 0.25
    pulse = komm.RootRaisedCosinePulse(rolloff=α)
    assert np.isclose(pulse.waveform(0.0), 1 + α * (4 / np.pi - 1))


@pytest.mark.parametrize("rolloff", [-0.1, 1.5])
def test_rrc_invalid_rolloff(rolloff):
    with pytest.raises(ValueError, match="must satisfy 0 <= rolloff <= 1"):
        komm.RootRaisedCosinePulse(rolloff=rolloff)
