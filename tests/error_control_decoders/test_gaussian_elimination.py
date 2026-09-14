from itertools import product

import numpy as np
import pytest

import komm
import komm.abc


@pytest.fixture(
    params=[
        komm.HammingCode(3),
        komm.BlockCode([[1, 0, 0, 1, 1], [0, 1, 1, 1, 0]]),
        komm.RepetitionCode(5),
        komm.SingleParityCheckCode(5),
        komm.CordaroWagnerCode(5),
        komm.ReedMullerCode(1, 4),
    ]
)
def code(request: pytest.FixtureRequest) -> komm.abc.BlockCode:
    return request.param


def exhaustive_bitwise_map_decoder(code: komm.abc.BlockCode, b):
    codewords = code.codewords()
    compat_v = codewords[np.all((b == 2) | (b == codewords), axis=1)]
    compat_u = code.inverse_encode(compat_v)
    v_hat = np.where(np.all(compat_v == compat_v[0], axis=0), compat_v[0], 2)
    u_hat = np.where(np.all(compat_u == compat_u[0], axis=0), compat_u[0], 2)
    return v_hat, u_hat


@pytest.mark.parametrize(
    "code", [komm.HammingCode(3), komm.BlockCode([[1, 0, 0, 1, 1], [0, 1, 1, 1, 0]])]
)
def test_gaussian_elimination_exhaustive(code: komm.abc.BlockCode):
    decoder = komm.GaussianEliminationDecoder(code)
    for u in product([0, 1], repeat=code.dimension):
        v = code.encode(u)
        for mask in product([False, True], repeat=code.length):
            b = np.where(mask, 2, v)
            v_hat, u_hat = exhaustive_bitwise_map_decoder(code, b)
            np.testing.assert_array_equal(decoder.decode(b), u_hat)


def test_gaussian_elimination_correct_bits(code: komm.abc.BlockCode):
    # Every returned bit matches the true message.
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.5)
    decoder = komm.GaussianEliminationDecoder(code)
    for _ in range(100):
        u = dms.emit(code.dimension)
        r = bec.transmit(code.encode(u))
        u_hat = decoder.decode(r)
        known = u_hat != 2
        np.testing.assert_equal(u_hat[known], u[known])


def test_gaussian_elimination_unique(code: komm.abc.BlockCode):
    # Fewer erasures than the minimum distance means unique decoding.
    dms = komm.DiscreteMemorylessSource(2)
    bec = komm.BinaryErasureChannel(0.3)
    decoder = komm.GaussianEliminationDecoder(code)
    for _ in range(100):
        u = dms.emit(code.dimension)
        r = bec.transmit(code.encode(u))
        if np.count_nonzero(r == 2) < code.minimum_distance():
            np.testing.assert_equal(decoder.decode(r), u)


def test_gaussian_elimination_no_erasures(code: komm.abc.BlockCode):
    # Without erasures, every message is recovered.
    decoder = komm.GaussianEliminationDecoder(code)
    k = code.dimension
    messages = komm.int_to_bits(range(2**k), width=k).reshape(-1, k)
    np.testing.assert_equal(decoder.decode(code.codewords()), messages)


@pytest.mark.parametrize(
    "r",
    [
        [1, 1, 0, 3, 0, 1, 1],
        [-1.3, -0.8, 1.1, -0.8, 1.2, -0.2, -1.4],
        [1.3, 0.8, 1.1, 0.8, 1.2, 0.2, 1.4],
    ],
)
def test_gaussian_elimination_invalid_input(r):
    # Only bits and erasures are accepted.
    decoder = komm.GaussianEliminationDecoder(komm.HammingCode(3))
    with pytest.raises(ValueError):
        decoder.decode(r)
