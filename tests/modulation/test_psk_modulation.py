import numpy as np
import pytest

import komm

w8 = (1.0 + 1.0j) / np.sqrt(2)
w8c = w8.conjugate()


@pytest.mark.parametrize(
    "order, bits_per_symbol, constellation, minimum_distance",
    [
        (2, 1, [1, -1], 2.0),
        (4, 2, [1, 1j, -1, -1j], np.sqrt(2)),
        (8, 3, [1, w8, 1j, -w8c, -1, -w8, -1j, w8c], np.sqrt(2 - np.sqrt(2))),
    ],
)
def test_psk_modulation_1(order, constellation, bits_per_symbol, minimum_distance):
    psk = komm.PSKModulation(order)
    assert psk.order == order
    assert psk.bits_per_symbol == bits_per_symbol
    assert np.allclose(psk.constellation, constellation)
    assert np.allclose(psk.energy_per_symbol, 1.0)
    assert np.allclose(psk.energy_per_bit, 1.0 / bits_per_symbol)
    assert np.allclose(psk.symbol_mean, 0.0)
    assert np.allclose(psk.minimum_distance, minimum_distance)


@pytest.mark.parametrize(
    "amplitude, phase_offset, constellation",
    [
        (1.0, 0.0, [1, 1j, -1, -1j]),
        (2.0, 0.0, [2, 2j, -2, -2j]),
        (3.0, np.pi / 4.0, np.array([3 * w8, -3 * w8c, -3 * w8, 3 * w8c])),
        (4.0, np.pi / 2.0, np.array([4j, -4, -4j, 4])),
        (5.0, np.pi, np.array([-5, -5j, 5, 5j])),
    ],
)
def test_psk_modulation_2(amplitude, phase_offset, constellation):
    qpsk = komm.PSKModulation(4, amplitude=amplitude, phase_offset=phase_offset)
    assert np.allclose(qpsk.constellation, constellation)
    assert np.allclose(qpsk.energy_per_symbol, amplitude**2)
    assert np.allclose(qpsk.energy_per_bit, 0.5 * amplitude**2)
    assert np.allclose(qpsk.symbol_mean, 0.0)
    assert np.allclose(qpsk.minimum_distance, amplitude * np.sqrt(2))


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("labeling", ["natural", "reflected"])
def test_psk_modulation_3(order, labeling):
    # Test hard demodulation
    psk = komm.PSKModulation(order, labeling=labeling)
    m = psk.bits_per_symbol
    bits = np.random.randint(0, 2, size=100 * m, dtype=int)
    assert np.allclose(psk.demodulate_hard(psk.modulate(bits)), bits)
