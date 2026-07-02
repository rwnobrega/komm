import numpy as np

import komm


def test_rrc_waveform_scalar_regression():
    α = 0.25
    pulse = komm.RootRaisedCosinePulse(rolloff=α)
    assert np.isclose(pulse.waveform(0.0), 1 + α * (4 / np.pi - 1))
