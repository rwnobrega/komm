import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "base_amplitude": 1, "phase_offset": 0},
            {
                "bits_per_symbol": 1,
                "constellation": [0, 1],
                "energy_per_symbol": 0.5,
                "energy_per_bit": 0.5,
                "symbol_mean": 0.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "base_amplitude": 1, "phase_offset": 0},
            {
                "bits_per_symbol": 2,
                "constellation": [0, 1, 2, 3],
                "energy_per_symbol": 3.5,
                "energy_per_bit": 1.75,
                "symbol_mean": 1.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 8, "base_amplitude": 1, "phase_offset": 0},
            {
                "bits_per_symbol": 3,
                "constellation": [0, 1, 2, 3, 4, 5, 6, 7],
                "energy_per_symbol": 17.5,
                "energy_per_bit": 35 / 6,
                "symbol_mean": 3.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "base_amplitude": 1, "phase_offset": 0},
            {
                "bits_per_symbol": 2,
                "constellation": [0, 1, 2, 3],
                "energy_per_symbol": 3.5,
                "energy_per_bit": 1.75,
                "symbol_mean": 1.5,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "base_amplitude": 2, "phase_offset": 0},
            {
                "bits_per_symbol": 2,
                "constellation": [0, 2, 4, 6],
                "energy_per_symbol": 14,
                "energy_per_bit": 7,
                "symbol_mean": 3,
                "minimum_distance": 2,
            },
        ),
        (
            {"order": 4, "base_amplitude": 1, "phase_offset": np.pi / 4},
            {
                "bits_per_symbol": 2,
                "constellation": np.array([0, 1 + 1j, 2 + 2j, 3 + 3j]) / np.sqrt(2),
                "energy_per_symbol": 3.5,
                "energy_per_bit": 1.75,
                "symbol_mean": (1.5 + 1.5j) / np.sqrt(2),
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "base_amplitude": 0.5, "phase_offset": np.pi},
            {
                "bits_per_symbol": 2,
                "constellation": [0, -0.5, -1, -1.5],
                "energy_per_symbol": 0.875,
                "energy_per_bit": 0.4375,
                "symbol_mean": -0.75,
                "minimum_distance": 0.5,
            },
        ),
        (
            {"order": 4, "base_amplitude": 2.5, "phase_offset": np.pi / 2},
            {
                "bits_per_symbol": 2,
                "constellation": [0, 2.5j, 5j, 7.5j],
                "energy_per_symbol": 21.875,
                "energy_per_bit": 10.9375,
                "symbol_mean": 3.75j,
                "minimum_distance": 2.5,
            },
        ),
    ],
)
def test_ask_parameters(params, expected):
    ask = komm.ASKModulation(**params)
    assert ask.order == params["order"]
    assert ask.bits_per_symbol == expected["bits_per_symbol"]
    np.testing.assert_allclose(ask.constellation, expected["constellation"])
    np.testing.assert_allclose(ask.energy_per_symbol, expected["energy_per_symbol"])
    np.testing.assert_allclose(ask.energy_per_bit, expected["energy_per_bit"])
    np.testing.assert_allclose(ask.symbol_mean, expected["symbol_mean"])
    np.testing.assert_allclose(ask.minimum_distance, expected["minimum_distance"])


@pytest.mark.parametrize(
    "order, demodulated",
    [
        (2, [0, 0, 0, 1, 1, 1]),
        (4, [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]),
        (8, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]),
    ],
)
def test_ask_demodulate_hard(order, demodulated):
    ask = komm.ASKModulation(order)
    y = [-0.5, 0.25, 0.4, 0.65, 2.1, 10]
    assert np.allclose(ask.demodulate_hard(y), demodulated)
