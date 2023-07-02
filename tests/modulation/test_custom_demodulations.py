import numpy as np
import pytest

import komm


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("labeling", ["natural", "reflected"])
@pytest.mark.parametrize("channel_snr", [0.3, 1.0, 3.0, 10.0])
def test_pam_demodulate_hard(order, labeling, channel_snr):
    pam = komm.PAModulation(order, labeling=labeling)
    channel = komm.AWGNChannel(signal_power=pam.energy_per_symbol, snr=channel_snr)
    m = pam.bits_per_symbol
    bits = np.random.randint(0, 2, size=100 * m, dtype=int)
    x = pam.modulate(bits)
    y = channel(x)
    assert np.array_equal(
        komm.Modulation._demodulate_hard(pam, y),
        komm.PAModulation._demodulate_hard(pam, y),
    )


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("labeling", ["natural", "reflected"])
@pytest.mark.parametrize("channel_snr", [0.3, 1.0, 3.0, 10.0])
def test_pam_demodulate_soft(order, labeling, channel_snr):
    pam = komm.PAModulation(order, labeling=labeling)
    channel = komm.AWGNChannel(signal_power=pam.energy_per_symbol, snr=channel_snr)
    m = pam.bits_per_symbol
    bits = np.random.randint(0, 2, size=100 * m, dtype=int)
    x = pam.modulate(bits)
    y = channel(x)
    assert np.allclose(
        komm.Modulation._demodulate_soft(pam, y, channel_snr=channel_snr),
        komm.PAModulation._demodulate_soft(pam, y, channel_snr=channel_snr),
    )
