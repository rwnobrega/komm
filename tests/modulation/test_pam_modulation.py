import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "order, bits_per_symbol, constellation, energy_per_symbol, energy_per_bit",
    [
        (2, 1, [-1, 1], 1.0, 1.0),
        (4, 2, [-3, -1, 1, 3], 5.0, 2.5),
        (8, 3, [-7, -5, -3, -1, 1, 3, 5, 7], 21.0, 7.0),
    ],
)
def test_pam_modulation_1(order, constellation, bits_per_symbol, energy_per_symbol, energy_per_bit):
    pam = komm.PAModulation(order)
    assert pam.order == order
    assert pam.bits_per_symbol == bits_per_symbol
    assert np.allclose(pam.constellation, constellation)
    assert np.allclose(pam.energy_per_symbol, energy_per_symbol)
    assert np.allclose(pam.energy_per_bit, energy_per_bit)
    assert np.allclose(pam.symbol_mean, 0.0)
    assert np.allclose(pam.minimum_distance, 2.0)


@pytest.mark.parametrize(
    "base_amplitude, constellation",
    [
        (1.0, [-7, -5, -3, -1, 1, 3, 5, 7]),
        (2.0, [-14, -10, -6, -2, 2, 6, 10, 14]),
        (0.5, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]),
    ],
)
def test_pam_modulation_2(base_amplitude, constellation):
    pam8 = komm.PAModulation(8, base_amplitude=base_amplitude)
    assert np.allclose(pam8.constellation, constellation)
    assert np.allclose(pam8.energy_per_symbol, 21.0 * base_amplitude**2)
    assert np.allclose(pam8.energy_per_bit, 7.0 * base_amplitude**2)
    assert np.allclose(pam8.symbol_mean, 0.0)
    assert np.allclose(pam8.minimum_distance, 2.0 * base_amplitude)


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("labeling", ["natural", "reflected"])
def test_pam_modulation_3(order, labeling):
    # Test hard demodulation
    pam = komm.PAModulation(order, labeling=labeling)
    m = pam.bits_per_symbol
    bits = np.random.randint(0, 2, size=100 * m, dtype=int)
    x = pam.modulate(bits)
    y = x
    assert np.allclose(pam.demodulate(y, decision_method="hard"), bits)


@pytest.mark.parametrize(
    "order, demodulated",
    [
        (2, [0, 0, 0, 1, 1, 1]),
        (4, [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]),
        (8, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]),
    ],
)
def test_pam_modulation_4(order, demodulated):
    pam = komm.PAModulation(order)
    y = [-20.0, -7.1, -0.5, 1.5, 6.2, 100.0]
    assert np.allclose(pam.demodulate(y), demodulated)


@pytest.mark.parametrize("channel_snr", [0.1, 0.3, 1.0, 3.0, 10.0, np.inf])
def test_pam_modulation_5(channel_snr):
    # Test soft demodulation for 2-PAM
    pam2 = komm.PAModulation(2)
    assert np.allclose(
        pam2.demodulate([-1.0, 1.0], decision_method="soft", channel_snr=channel_snr),
        [4.0 * channel_snr, -4.0 * channel_snr],
    )
