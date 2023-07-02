import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "order, bits_per_symbol, constellation, energy_per_symbol, energy_per_bit, symbol_mean",
    [
        (2, 1, [0.0, 1.0], 0.5, 0.5, 0.5),
        (4, 2, [0.0, 1.0, 2.0, 3.0], 3.5, 1.75, 1.5),
        (8, 3, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 17.5, 35 / 6, 3.5),
    ],
)
def test_ask_modulation_1(order, constellation, bits_per_symbol, energy_per_symbol, energy_per_bit, symbol_mean):
    ask = komm.ASKModulation(order)
    assert ask.order == order
    assert ask.bits_per_symbol == bits_per_symbol
    assert np.allclose(ask.constellation, constellation)
    assert np.allclose(ask.energy_per_symbol, energy_per_symbol)
    assert np.allclose(ask.energy_per_bit, energy_per_bit)
    assert np.allclose(ask.symbol_mean, symbol_mean)
    assert np.allclose(ask.minimum_distance, 1.0)


@pytest.mark.parametrize(
    "base_amplitude, phase_offset, constellation",
    [
        (1.0, 0.0, [0.0, 1.0, 2.0, 3.0]),
        (2.0, 0.0, [0.0, 2.0, 4.0, 6.0]),
        (1.0, np.pi / 4.0, np.array([0, 1 + 1j, 2 + 2j, 3 + 3j]) / np.sqrt(2)),
        (0.5, np.pi, [0, -0.5, -1, -1.5]),
        (2.5, np.pi / 2.0, [0, 2.5j, 5j, 7.5j]),
    ],
)
def test_ask_modulation_2(base_amplitude, phase_offset, constellation):
    ask4 = komm.ASKModulation(4, base_amplitude=base_amplitude, phase_offset=phase_offset)
    assert np.allclose(ask4.constellation, constellation)
    assert np.allclose(ask4.energy_per_symbol, 3.5 * base_amplitude**2)
    assert np.allclose(ask4.energy_per_bit, 1.75 * base_amplitude**2)
    assert np.allclose(ask4.symbol_mean, 1.5 * base_amplitude * np.exp(1j * phase_offset))
    assert np.allclose(ask4.minimum_distance, base_amplitude)


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("labeling", ["natural", "reflected"])
def test_ask_modulation_3(order, labeling):
    # Test hard demodulation
    ask = komm.PAModulation(order, labeling=labeling)
    m = ask.bits_per_symbol
    bits = np.random.randint(0, 2, size=100 * m, dtype=int)
    assert np.allclose(ask.demodulate(ask.modulate(bits)), bits)


@pytest.mark.parametrize(
    "order, demodulated",
    [
        (2, [0, 0, 0, 1, 1, 1]),
        (4, [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]),
        (8, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]),
    ],
)
def test_ask_modulation_4(order, demodulated):
    ask = komm.ASKModulation(order)
    y = [-0.5, 0.25, 0.4, 0.65, 2.1, 10.0]
    assert np.allclose(ask.demodulate(y), demodulated)
