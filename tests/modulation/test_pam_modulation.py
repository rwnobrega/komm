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
    np.testing.assert_allclose(pam.labeling, [[0, 0], [0, 1], [1, 1], [1, 0]])
    pam = komm.PAModulation(4, labeling="natural")
    np.testing.assert_allclose(pam.labeling, [[0, 0], [0, 1], [1, 0], [1, 1]])


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
        (4, [-3, 1, -3, 3, 3, -1]),
        (8, [-5, 7, 5, -5]),
    ],
)
def test_pam_modulate(order, modulated):
    pam = komm.PAModulation(order)
    bits = [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    np.testing.assert_array_equal(pam.modulate(bits), modulated)


@pytest.mark.parametrize(
    "order, demodulated",
    [
        (2, [0, 0, 0, 1, 1, 1]),
        (4, [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0]),
        (8, [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0]),
    ],
)
def test_pam_demodulate_hard(order, demodulated):
    pam = komm.PAModulation(order)
    received = [-20.0, -7.1, -0.5, 1.5, 6.2, 100.0]
    np.testing.assert_allclose(pam.demodulate_hard(received), demodulated)
