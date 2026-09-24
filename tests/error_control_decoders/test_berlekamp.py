import numpy as np
import pytest

import komm
from komm._algebra.bifield import horner, power
from komm._error_control_decoders.BerlekampDecoder import (
    berlekamp_algorithm,
    forney_algorithm,
)


def test_berlekamp_bch_lin_costello():
    # [LC04, Example 6.5]
    code = komm.BCHCode(mu=4, delta=7)
    decoder = komm.BerlekampDecoder(code)
    field = code.field
    alpha = code.alpha
    r = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    points = power(field, int(alpha), range(1, code.delta))
    syndrome = horner(field, r, points)
    assert np.array_equal(syndrome, power(field, int(alpha), [0, 0, 10, 0, 10, 5]))
    sigma = berlekamp_algorithm(field, syndrome)
    assert np.array_equal(sigma, [1, 1, 0, int(alpha**5)])
    inverses = power(field, int(alpha), -np.arange(code.length))
    e_loc = np.flatnonzero(horner(field, sigma, inverses) == 0)
    assert np.array_equal(e_loc, [3, 5, 12])
    roots = inverses[e_loc]
    assert np.array_equal(roots, [int(alpha**12), int(alpha**10), int(alpha**3)])
    u_hat = decoder.decode(r)
    assert np.array_equal(u_hat, [0, 0, 0, 0, 0])


codes = [
    komm.BCHCode(2, 3),
    komm.BCHCode(3, 3),
    komm.BCHCode(3, 7),
    komm.BCHCode(4, 3),
    komm.BCHCode(4, 5),
    komm.BCHCode(4, 7),
    komm.BCHCode(4, 15),
    komm.ReedSolomonCode(2, 3),
    komm.ReedSolomonCode(3, 4),
    komm.ReedSolomonCode(3, 5),
    komm.ReedSolomonCode(4, 7),
    komm.ReedSolomonCode(4, 10),
    komm.ReedSolomonCode(5, 11),
]


def random_error_pattern(code: komm.BCHCode | komm.ReedSolomonCode, w: int):
    # Nonzero values at w random symbols.
    width = code.mu if isinstance(code, komm.ReedSolomonCode) else 1
    n = 2**code.mu - 1
    e = np.zeros(n, dtype=int)
    e[np.random.choice(n, w, replace=False)] = np.random.randint(1, 2**width, w)
    return komm.int_to_bits(e, width=width)


@pytest.mark.parametrize("code", codes)
def test_berlekamp_error_correcting_capability(code):
    t = (code.delta - 1) // 2
    decoder = komm.BerlekampDecoder(code)
    for w in range(t + 1):
        for _ in range(10):
            u = np.random.randint(0, 2, code.dimension)
            r = code.encode(u) ^ random_error_pattern(code, w)
            assert np.array_equal(decoder.decode(r), u)


@pytest.mark.parametrize("code", codes)
def test_berlekamp_above_error_correcting_capability(code):
    n, t = 2**code.mu - 1, (code.delta - 1) // 2
    decoder = komm.BerlekampDecoder(code)
    for w in range(t + 1, n + 1):
        for _ in range(10):
            r = random_error_pattern(code, w)
            v_hat = decoder.decode_to_codeword(r)
            # Either a codeword or the received word.
            assert not np.any(code.check(v_hat)) or np.array_equal(v_hat, r)


@pytest.mark.parametrize(
    "mu, delta",
    [(2, 3), (3, 4), (3, 5), (4, 7), (4, 10), (6, 11)],
)
def test_forney_error_values(mu, delta):
    code = komm.ReedSolomonCode(mu, delta)
    field, alpha = code.field, int(code.alpha)
    n, t = 2**mu - 1, (delta - 1) // 2
    points = power(field, alpha, range(1, delta))
    inverses = power(field, alpha, -np.arange(n))
    for w in range(1, t + 1):
        for _ in range(10):
            e = np.zeros(n, dtype=int)
            e_loc = np.sort(np.random.choice(n, w, replace=False))
            e[e_loc] = np.random.randint(1, 2**mu, w)
            syndrome = horner(field, e, points)
            sigma = berlekamp_algorithm(field, syndrome)
            values = forney_algorithm(field, syndrome, sigma, inverses[e_loc])
            assert np.array_equal(values, e[e_loc])
