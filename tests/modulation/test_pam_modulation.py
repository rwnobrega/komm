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
def test_pam_modulation_order(
    order, constellation, bits_per_symbol, energy_per_symbol, energy_per_bit
):
    pam = komm.PAModulation(order)
    assert pam.order == order
    assert pam.bits_per_symbol == bits_per_symbol
    np.testing.assert_allclose(pam.constellation, constellation)
    np.testing.assert_allclose(pam.energy_per_symbol, energy_per_symbol)
    np.testing.assert_allclose(pam.energy_per_bit, energy_per_bit)
    np.testing.assert_allclose(pam.symbol_mean, 0.0)
    np.testing.assert_allclose(pam.minimum_distance, 2.0)


@pytest.mark.parametrize(
    "base_amplitude, constellation",
    [
        (1.0, [-7, -5, -3, -1, 1, 3, 5, 7]),
        (2.0, [-14, -10, -6, -2, 2, 6, 10, 14]),
        (0.5, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]),
    ],
)
def test_pam_base_amplitude(base_amplitude, constellation):
    pam8 = komm.PAModulation(8, base_amplitude=base_amplitude)
    np.testing.assert_allclose(pam8.constellation, constellation)
    np.testing.assert_allclose(pam8.energy_per_symbol, 21.0 * base_amplitude**2)
    np.testing.assert_allclose(pam8.energy_per_bit, 7.0 * base_amplitude**2)
    np.testing.assert_allclose(pam8.symbol_mean, 0.0)
    np.testing.assert_allclose(pam8.minimum_distance, 2.0 * base_amplitude)


def test_pam_labeling():
    pam = komm.PAModulation(4, labeling="reflected")
    np.testing.assert_allclose(pam.labeling, [[0, 0], [1, 0], [1, 1], [0, 1]])
    pam = komm.PAModulation(4, labeling="natural")
    np.testing.assert_allclose(pam.labeling, [[0, 0], [1, 0], [0, 1], [1, 1]])


def test_pam_invalid():
    with pytest.raises(ValueError):  # Invalid order
        komm.PAModulation(3)
    with pytest.raises(ValueError):  # Invalid labeling
        komm.PAModulation(4, labeling="invalid")
    with pytest.raises(ValueError):  # Invalid labeling
        komm.PAModulation(4, labeling=[[0, 0], [1, 0], [1, 1]])


@pytest.mark.parametrize(
    "order, modulated",
    [
        (2, [-1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1]),
        (4, [-3, 1, -3, -1, -1, 3]),
        (8, [7, -5, 5, 7]),
    ],
)
def test_pam_modulatate(order, modulated):
    pam = komm.PAModulation(order)
    bits = [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    np.testing.assert_array_equal(pam.modulate(bits), modulated)


@pytest.mark.parametrize(
    "order, demodulated",
    [
        (2, [0, 0, 0, 1, 1, 1]),
        (4, [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]),
        (8, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]),
    ],
)
def test_pam_demodulate_hard(order, demodulated):
    pam = komm.PAModulation(order)
    received = [-20.0, -7.1, -0.5, 1.5, 6.2, 100.0]
    np.testing.assert_allclose(pam.demodulate_hard(received), demodulated)


@pytest.mark.parametrize("snr", [0.1, 0.3, 1.0, 3.0, 10.0, np.inf])
def test_pam_demodulate_soft(snr):
    # Test soft demodulation for 2-PAM
    pam2 = komm.PAModulation(2)
    np.testing.assert_allclose(
        pam2.demodulate_soft([-1.0, 1.0], snr=snr),
        [4.0 * snr, -4.0 * snr],
    )


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("labeling", ["natural", "reflected"])
def test_pam_modem(order, labeling):
    pam = komm.PAModulation(order, labeling=labeling)
    m = pam.bits_per_symbol
    bits = np.random.randint(0, 2, size=100 * m, dtype=int)
    symbols = pam.modulate(bits)
    bits_hat_hard = pam.demodulate_hard(symbols)
    np.testing.assert_allclose(bits_hat_hard, bits)
    bits_hat_soft = (pam.demodulate_soft(symbols, snr=1000.0) < 0).astype(int)
    np.testing.assert_allclose(bits_hat_soft, bits)


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("labeling", ["natural", "reflected"])
@pytest.mark.parametrize("snr", [0.3, 1.0, 3.0, 10.0])
def test_pam_general_modulation_equivalent(order, labeling, snr):
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
