from itertools import product

import numpy as np
import pytest

import komm
import komm.abc

params = []

# PAM
order = [2, 4, 8]
base_amplitude = [0.5, 1.0, 2.0]
labeling = ["natural", "reflected"]
for args in product(order, base_amplitude, labeling):
    params.append(komm.PAModulation(*args))

# QAM
orders = [4, 16, (2, 4), (8, 2)]
base_amplitudes = [0.5, 2.0, (0.5, 1.0)]
phase_offset = [0.0, np.pi / 4]
labeling = ["natural_2d", "reflected_2d"]
for args in product(orders, base_amplitudes, phase_offset, labeling):
    params.append(komm.QAModulation(*args))

# PSK
order = [2, 4, 8, 16]
amplitude = [0.5, 1.0, 2.0]
phase_offset = [0.0, np.pi / 4, np.pi / 3]
labeling = ["natural", "reflected"]
for args in product(order, amplitude, phase_offset, labeling):
    params.append(komm.PSKModulation(*args))


@pytest.fixture(params=params, ids=lambda mod: repr(mod))
def mod(request: pytest.FixtureRequest):
    return request.param


def test_equivalence_properties(mod: komm.abc.Modulation):
    ref = komm.Modulation(mod.constellation, mod.labeling)
    np.testing.assert_almost_equal(ref.constellation, mod.constellation)
    np.testing.assert_equal(ref.labeling, mod.labeling)
    np.testing.assert_equal(ref.inverse_labeling, mod.inverse_labeling)
    np.testing.assert_equal(ref.order, mod.order)
    np.testing.assert_almost_equal(ref.bits_per_symbol, mod.bits_per_symbol)
    np.testing.assert_almost_equal(ref.energy_per_symbol, mod.energy_per_symbol)
    np.testing.assert_almost_equal(ref.energy_per_bit, mod.energy_per_bit)
    np.testing.assert_almost_equal(ref.symbol_mean, mod.symbol_mean)
    np.testing.assert_almost_equal(ref.minimum_distance, mod.minimum_distance)


@pytest.mark.parametrize("snr", [0.3, 1.0, 3.0, 10.0], ids=lambda x: f"snr={x}")
def test_equivalence_modulate_demodulate(mod: komm.abc.Modulation, snr):
    ref = komm.Modulation(mod.constellation, mod.labeling)
    channel = komm.AWGNChannel(signal_power=mod.energy_per_symbol, snr=snr)
    bits = np.random.randint(0, 2, size=100 * mod.bits_per_symbol, dtype=int)
    symbols = mod.modulate(bits)
    received = channel(symbols)

    symbols1 = ref.modulate(bits)
    np.testing.assert_array_almost_equal(symbols, symbols1)

    demodulated_hard = mod.demodulate_hard(received)
    demodulated_hard1 = ref.demodulate_hard(received)
    np.testing.assert_array_almost_equal(demodulated_hard, demodulated_hard1)

    demodulated_soft = mod.demodulate_soft(received, snr)
    demodulated_soft1 = ref.demodulate_soft(received, snr)
    np.testing.assert_array_almost_equal(demodulated_soft, demodulated_soft1)


def test_modulation_high_snr(mod: komm.abc.Modulation):
    ref = komm.Modulation(mod.constellation, mod.labeling)
    bits = np.random.randint(0, 2, size=100 * mod.bits_per_symbol, dtype=int)
    symbols = mod.modulate(bits)
    np.testing.assert_array_equal(
        mod.demodulate_hard(symbols),
        bits,
    )
    np.testing.assert_array_equal(
        ref.demodulate_hard(symbols),
        bits,
    )
    np.testing.assert_array_equal(
        (mod.demodulate_soft(symbols, snr=1e4) < 0).astype(int),
        bits,
    )
    np.testing.assert_array_equal(
        (ref.demodulate_soft(symbols, snr=1e4) < 0).astype(int),
        bits,
    )
