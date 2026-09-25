from itertools import product

import numpy as np
import pytest

import komm
import komm.abc
from komm._error_control_decoders.GaussianEliminationDecoder import (
    _codeword_from_erased,
    _codeword_from_known,
    _message_from_erased,
    _message_from_known,
)

small_codes = [
    komm.HammingCode(3),
    komm.BlockCode([[1, 0, 0, 1, 1], [0, 1, 1, 1, 0]]),
    komm.BlockCode([[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]]),
    komm.BlockCode([[1, 0, 1, 1, 1], [1, 1, 0, 0, 1], [0, 1, 1, 0, 1]]),
]

all_codes = small_codes + [
    komm.RepetitionCode(5),
    komm.SingleParityCheckCode(5),
    komm.CordaroWagnerCode(5),
    komm.ReedMullerCode(1, 4),
]


def exhaustive_bitwise_map_decoder(code: komm.abc.BlockCode, b):
    codewords = code.codewords()
    compat_v = codewords[np.all((b == 2) | (b == codewords), axis=1)]
    compat_u = code.inverse_encode(compat_v)
    v_hat = np.where(np.all(compat_v == compat_v[0], axis=0), compat_v[0], 2)
    u_hat = np.where(np.all(compat_u == compat_u[0], axis=0), compat_u[0], 2)
    return v_hat, u_hat


@pytest.mark.parametrize("code", small_codes)
def test_gaussian_elimination_exhaustive(code: komm.abc.BlockCode):
    decoder = komm.GaussianEliminationDecoder(code)
    for u in product([0, 1], repeat=code.dimension):
        v = code.encode(u)
        for mask in product([False, True], repeat=code.length):
            b = np.where(mask, 2, v)
            v_hat, u_hat = exhaustive_bitwise_map_decoder(code, b)
            np.testing.assert_array_equal(decoder.decode(b), u_hat)
            np.testing.assert_array_equal(decoder.decode_to_codeword(b), v_hat)


@pytest.mark.parametrize("code", small_codes)
def test_gaussian_elimination_both_paths_agree(code: komm.abc.BlockCode):
    G, H = code.generator_matrix, code.check_matrix
    G_r_inv = code.generator_matrix_right_inverse
    for v in code.codewords():
        for mask in product([False, True], repeat=code.length):
            r = np.where(mask, 2, v)
            np.testing.assert_array_equal(
                _codeword_from_erased(H, r),
                _codeword_from_known(G, r),
            )
            np.testing.assert_array_equal(
                _message_from_erased(H, G_r_inv, r),
                _message_from_known(G, r),
            )


@pytest.mark.parametrize("code", all_codes)
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


@pytest.mark.parametrize("code", all_codes)
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


@pytest.mark.parametrize("code", all_codes)
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
