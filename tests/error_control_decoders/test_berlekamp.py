import numpy as np
import pytest

import komm
from komm._algebra.FiniteBifield import horner
from komm._error_control_decoders.BerlekampDecoder import berlekamp_algorithm


def test_berlekamp_lin_costello():
    # [LC04, Example 6.5]
    code = komm.BCHCode(mu=4, delta=7)
    decoder = komm.BerlekampDecoder(code)
    field = code.field
    alpha = code.alpha
    r = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    points = [int(alpha**i) for i in range(1, code.delta)]
    syndrome = horner(field, r, points)
    assert np.array_equal(
        syndrome,
        [1, 1, int(alpha**10), 1, int(alpha**10), int(alpha**5)],
    )
    sigma = berlekamp_algorithm(field, syndrome)
    assert np.array_equal(sigma, [1, 1, 0, int(alpha**5)])
    inverses = field.power(int(alpha), -np.arange(code.length))
    e_loc = np.flatnonzero(horner(field, sigma, inverses) == 0)
    assert np.array_equal(e_loc, [3, 5, 12])
    roots = inverses[e_loc]
    assert np.array_equal(roots, [int(alpha**12), int(alpha**10), int(alpha**3)])
    u_hat = decoder.decode(r)
    assert np.array_equal(u_hat, [0, 0, 0, 0, 0])


@pytest.mark.parametrize("mu, deltas", [(2, [3]), (3, [3, 7]), (4, [3, 5, 7, 15])])
def test_berlekamp_error_correcting_capability(mu, deltas):
    for delta in deltas:
        code = komm.BCHCode(mu, delta)
        k, n = code.dimension, code.length
        decoder = komm.BerlekampDecoder(code)
        for w in range((delta - 1) // 2 + 1):
            for _ in range(10):
                r = np.zeros(n, dtype=int)
                error_locations = np.random.choice(n, w, replace=False)
                r[error_locations] ^= 1
                assert np.array_equal(decoder.decode(r), np.zeros(k, dtype=int))


def test_berlekamp_above_error_correcting_capability():
    code = komm.BCHCode(mu=4, delta=7)
    n = code.length
    t = (code.delta - 1) // 2
    decoder = komm.BerlekampDecoder(code)
    for w in range(t + 1, n + 1):
        for _ in range(10):
            r = np.zeros(code.length, dtype=int)
            error_locations = np.random.choice(code.length, w, replace=False)
            r[error_locations] ^= 1
            v_hat = decoder.decode_to_codeword(r)
            # Either a codeword or the received word.
            assert not np.any(code.check(v_hat)) or np.array_equal(v_hat, r)
