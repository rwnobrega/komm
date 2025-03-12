import numpy as np
import pytest

import komm

w8 = (1 + 1j) / np.sqrt(2)
w8c = np.conj(w8)


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"order": 2, "amplitude": 1, "phase_offset": 0},
            {
                "bits_per_symbol": 1,
                "constellation": [1, -1],
                "energy_per_symbol": 1,
                "energy_per_bit": 1,
                "minimum_distance": 2,
            },
        ),
        (
            {"order": 2, "amplitude": 0.5, "phase_offset": np.pi / 2},
            {
                "bits_per_symbol": 1,
                "constellation": [0.5j, -0.5j],
                "energy_per_symbol": 0.25,
                "energy_per_bit": 0.25,
                "minimum_distance": 1,
            },
        ),
        (
            {"order": 4, "amplitude": 1, "phase_offset": 0},
            {
                "bits_per_symbol": 2,
                "constellation": [1, 1j, -1, -1j],
                "energy_per_symbol": 1,
                "energy_per_bit": 0.5,
                "minimum_distance": np.sqrt(2),
            },
        ),
        (
            {"order": 4, "amplitude": 2, "phase_offset": np.pi / 2},
            {
                "bits_per_symbol": 2,
                "constellation": [2j, -2, -2j, 2],
                "energy_per_symbol": 4,
                "energy_per_bit": 2,
                "minimum_distance": 2 * np.sqrt(2),
            },
        ),
        (
            {"order": 4, "amplitude": 3, "phase_offset": np.pi / 4},
            {
                "bits_per_symbol": 2,
                "constellation": [3 * w8, -3 * w8c, -3 * w8, 3 * w8c],
                "energy_per_symbol": 9,
                "energy_per_bit": 4.5,
                "minimum_distance": 3 * np.sqrt(2),
            },
        ),
        (
            {"order": 8, "amplitude": 1, "phase_offset": 0},
            {
                "bits_per_symbol": 3,
                "constellation": [1, w8, 1j, -w8c, -1, -w8, -1j, w8c],
                "energy_per_symbol": 1,
                "energy_per_bit": 1 / 3,
                "minimum_distance": np.sqrt(2 - np.sqrt(2)),
            },
        ),
    ],
)
def test_psk_modulation(params, expected):
    psk = komm.PSKModulation(**params)
    assert psk.order == params["order"]
    assert psk.bits_per_symbol == expected["bits_per_symbol"]
    np.testing.assert_allclose(psk.constellation, expected["constellation"])
    np.testing.assert_allclose(psk.energy_per_symbol, expected["energy_per_symbol"])
    np.testing.assert_allclose(psk.energy_per_bit, expected["energy_per_bit"])
    np.testing.assert_allclose(psk.symbol_mean, 0)
    np.testing.assert_allclose(psk.minimum_distance, expected["minimum_distance"])
