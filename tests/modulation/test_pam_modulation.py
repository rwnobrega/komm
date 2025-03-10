import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "base_amplitude": 1.0},
            {
                "bits_per_symbol": 1,
                "constellation": [-1, 1],
                "energy_per_symbol": 1.0,
                "energy_per_bit": 1.0,
                "minimum_distance": 2.0,
            },
        ),
        (
            {"order": 4, "base_amplitude": 1.0},
            {
                "bits_per_symbol": 2,
                "constellation": [-3, -1, 1, 3],
                "energy_per_symbol": 5.0,
                "energy_per_bit": 2.5,
                "minimum_distance": 2.0,
            },
        ),
        (
            {"order": 8, "base_amplitude": 1.0},
            {
                "bits_per_symbol": 3,
                "constellation": [-7, -5, -3, -1, 1, 3, 5, 7],
                "energy_per_symbol": 21.0,
                "energy_per_bit": 7.0,
                "minimum_distance": 2.0,
            },
        ),
        (
            {"order": 8, "base_amplitude": 2.0},
            {
                "bits_per_symbol": 3,
                "constellation": [-14, -10, -6, -2, 2, 6, 10, 14],
                "energy_per_symbol": 84.0,
                "energy_per_bit": 28.0,
                "minimum_distance": 4.0,
            },
        ),
        (
            {"order": 8, "base_amplitude": 0.5},
            {
                "bits_per_symbol": 3,
                "constellation": [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
                "energy_per_symbol": 5.25,
                "energy_per_bit": 1.75,
                "minimum_distance": 1.0,
            },
        ),
    ],
)
def test_pam_parameters(params, expected):
    pam = komm.PAModulation(**params)
    assert pam.order == params["order"]
    assert pam.bits_per_symbol == expected["bits_per_symbol"]
    np.testing.assert_allclose(pam.constellation, expected["constellation"])
    np.testing.assert_allclose(pam.energy_per_symbol, expected["energy_per_symbol"])
    np.testing.assert_allclose(pam.energy_per_bit, expected["energy_per_bit"])
    np.testing.assert_allclose(pam.symbol_mean, 0.0)
    np.testing.assert_allclose(pam.minimum_distance, expected["minimum_distance"])


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
