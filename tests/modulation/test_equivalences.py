import numpy as np
import pytest

import komm


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("labeling", ["natural", "reflected"])
@pytest.mark.parametrize("snr", [0.3, 1.0, 3.0, 10.0])
def test_pam_equivalence(order, labeling, snr):
    pam = komm.PAModulation(order, labeling=labeling)
    mod = komm.Modulation(pam.constellation, pam.labeling)
    channel = komm.AWGNChannel(signal_power=pam.energy_per_symbol, snr=snr)

    np.testing.assert_array_equal(pam.constellation, mod.constellation)
    np.testing.assert_array_equal(pam.labeling, mod.labeling)
    assert pam.inverse_labeling == mod.inverse_labeling
    assert pam.bits_per_symbol == mod.bits_per_symbol
    assert pam.energy_per_symbol == mod.energy_per_symbol
    assert pam.energy_per_bit == mod.energy_per_bit
    assert pam.symbol_mean == mod.symbol_mean
    assert pam.minimum_distance == mod.minimum_distance

    bits = np.random.randint(0, 2, size=100 * pam.bits_per_symbol, dtype=int)
    symbols = pam.modulate(bits)
    received = channel(symbols)
    hard_bits = pam.demodulate_hard(received)
    soft_bits = pam.demodulate_soft(received, snr=snr)

    np.testing.assert_array_almost_equal(symbols, mod.modulate(bits))
    np.testing.assert_array_almost_equal(hard_bits, mod.demodulate_hard(received))
    np.testing.assert_array_almost_equal(soft_bits, mod.demodulate_soft(received, snr))
