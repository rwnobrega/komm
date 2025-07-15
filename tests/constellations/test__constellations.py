from itertools import product

import numpy as np
import pytest

import komm
import komm.abc

params = []

# PAM
order = [2, 4, 8]
base_amplitude = [0.5, 1.0, 2.0]
for args in product(order, base_amplitude):
    params.append(komm.PAMConstellation(*args))

# QAM
orders = [4, 16, (2, 4), (8, 2)]
base_amplitudes = [0.5, 2.0, (0.5, 1.0)]
phase_offset = [0.0, 1 / 8]
for args in product(orders, base_amplitudes, phase_offset):
    params.append(komm.QAMConstellation(*args))

# PSK
order = [2, 4, 8, 16]
amplitude = [0.5, 1.0, 2.0]
phase_offset = [0.0, 1 / 8, 1 / 6]
for args in product(order, amplitude, phase_offset):
    params.append(komm.PSKConstellation(*args))

# ASK
order = [2, 4, 8]
base_amplitude = [0.5, 1.0, 2.0]
phase_offset = [0.0, np.pi / 4, np.pi / 3]
for args in product(order, base_amplitude, phase_offset):
    params.append(komm.ASKConstellation(*args))


@pytest.fixture(params=params, ids=lambda const: repr(const))
def const(request: pytest.FixtureRequest):
    return request.param


def test_constellation_equivalence_properties(const: komm.abc.Constellation):
    ref = komm.Constellation(const.matrix)
    np.testing.assert_allclose(ref.matrix, const.matrix)
    np.testing.assert_equal(ref.order, const.order)
    np.testing.assert_equal(ref.dimension, const.dimension)
    np.testing.assert_allclose(ref.mean(), const.mean(), atol=1e-12)
    np.testing.assert_allclose(ref.mean_energy(), const.mean_energy())
    np.testing.assert_allclose(ref.minimum_distance(), const.minimum_distance())


@pytest.mark.parametrize("snr", [0.3, 1.0, 3.0, 10.0], ids=lambda x: f"snr={x}")
def test_constellation_equivalence_mod_demod(const: komm.abc.Constellation, snr):
    ref = komm.Constellation(const.matrix)
    channel = komm.AWGNChannel(signal_power=float(const.mean_energy()), snr=snr)
    indices = np.random.randint(0, const.order, size=100)
    received = channel.transmit(const.indices_to_symbols(indices))
    np.testing.assert_allclose(
        const.indices_to_symbols(indices),
        ref.indices_to_symbols(indices),
    )
    np.testing.assert_allclose(
        const.closest_indices(received),
        ref.closest_indices(received),
    )
    np.testing.assert_allclose(
        const.posteriors(received, snr),
        ref.posteriors(received, snr),
        atol=1e-12,
    )


def test_constellation_equivalence_high_snr(const: komm.abc.Constellation):
    ref = komm.Constellation(const.matrix)
    indices = np.random.randint(0, const.order, size=100)
    symbols = const.indices_to_symbols(indices)
    np.testing.assert_equal(const.closest_indices(symbols), indices)
    np.testing.assert_equal(ref.closest_indices(symbols), indices)
    np.testing.assert_equal(
        np.argmax(const.posteriors(symbols, snr=1e4).reshape(-1, const.order), axis=-1),
        indices,
    )
    np.testing.assert_equal(
        np.argmax(ref.posteriors(symbols, snr=1e4).reshape(-1, ref.order), axis=-1),
        indices,
    )
