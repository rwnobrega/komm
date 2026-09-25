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
    field, n = code.field, 2**code.mu - 1
    a = power(field, int(code.alpha), np.arange(n))  # a[i] = α^i
    r = np.zeros(n, dtype=int)
    r[[3, 5, 12]] = 1
    syndrome = horner(field, r, a[1 : code.delta])
    assert np.array_equal(syndrome, a[[0, 0, 10, 0, 10, 5]])
    sigma = berlekamp_algorithm(field, syndrome)
    assert np.array_equal(sigma, [a[0], a[0], 0, a[5]])
    inverses = a[-np.arange(n)]
    e_loc = np.flatnonzero(horner(field, sigma, inverses) == 0)
    assert np.array_equal(e_loc, [3, 5, 12])
    e_val = forney_algorithm(field, syndrome, sigma, inverses[e_loc])
    assert np.array_equal(e_val, [1, 1, 1])
    v_hat = decoder.decode_to_codeword(r)
    assert np.array_equal(v_hat, np.zeros(code.length, dtype=int))


def test_berlekamp_reed_solomon_lin_costello():
    # [LC04, Examples 7.2 and 7.3]
    code = komm.ReedSolomonCode(mu=4, delta=7)
    decoder = komm.BerlekampDecoder(code)
    field, n = code.field, 2**code.mu - 1
    a = power(field, int(code.alpha), np.arange(n))  # a[i] = α^i
    r = np.zeros(n, dtype=int)
    r[[3, 6, 12]] = a[[7, 3, 4]]
    syndrome = horner(field, r, a[1 : code.delta])
    assert np.array_equal(syndrome, [a[12], a[0], a[14], a[10], 0, a[12]])
    sigma = berlekamp_algorithm(field, syndrome)
    assert np.array_equal(sigma, a[[0, 7, 4, 6]])
    inverses = a[-np.arange(n)]
    e_loc = np.flatnonzero(horner(field, sigma, inverses) == 0)
    assert np.array_equal(e_loc, [3, 6, 12])
    e_val = forney_algorithm(field, syndrome, sigma, inverses[e_loc])
    assert np.array_equal(e_val, a[[7, 3, 4]])
    v_hat = decoder.decode_to_codeword(komm.int_to_bits(r, width=code.mu))
    assert np.array_equal(v_hat, np.zeros(code.length, dtype=int))


def test_berlekamp_reed_solomon_mceliece():
    # [McE04, Example 9.8]
    code = komm.ReedSolomonCode(mu=3, delta=5)
    decoder = komm.BerlekampDecoder(code)
    field, n = code.field, 2**code.mu - 1
    a = power(field, int(code.alpha), np.arange(n))  # a[i] = α^i
    r = np.array([a[3], a[1], a[0], a[2], 0, a[3], a[0]])
    syndrome = horner(field, r, a[1 : code.delta])
    assert np.array_equal(syndrome, [a[3], a[4], a[4], 0])
    sigma = berlekamp_algorithm(field, syndrome)
    assert np.array_equal(sigma, a[[0, 5, 5]])
    inverses = a[-np.arange(n)]
    e_loc = np.flatnonzero(horner(field, sigma, inverses) == 0)
    assert np.array_equal(e_loc, [2, 3])
    e_val = forney_algorithm(field, syndrome, sigma, inverses[e_loc])
    assert np.array_equal(e_val, a[[3, 6]])
    v_hat = decoder.decode_to_codeword(komm.int_to_bits(r, width=code.mu))
    c = [a[3], a[1], a[1], a[0], 0, a[3], a[0]]
    assert np.array_equal(v_hat, komm.int_to_bits(c, width=code.mu))


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


def random_error_pattern(
    rng: np.random.Generator, code: komm.BCHCode | komm.ReedSolomonCode, w: int
):
    # Nonzero values at w random symbols.
    width = code.mu if isinstance(code, komm.ReedSolomonCode) else 1
    n = 2**code.mu - 1
    e = np.zeros(n, dtype=int)
    e[rng.choice(n, w, replace=False)] = rng.integers(1, 2**width, w)
    return komm.int_to_bits(e, width=width)


@pytest.mark.parametrize("code", codes)
def test_berlekamp_error_correcting_capability(code, rng):
    t = (code.delta - 1) // 2
    decoder = komm.BerlekampDecoder(code)
    for w in range(t + 1):
        for _ in range(10):
            u = rng.integers(0, 2, code.dimension)
            r = code.encode(u) ^ random_error_pattern(rng, code, w)
            assert np.array_equal(decoder.decode(r), u)


@pytest.mark.parametrize("code", codes)
def test_berlekamp_above_error_correcting_capability(code, rng):
    n, t = 2**code.mu - 1, (code.delta - 1) // 2
    decoder = komm.BerlekampDecoder(code)
    for w in range(t + 1, n + 1):
        for _ in range(10):
            r = random_error_pattern(rng, code, w)
            v_hat = decoder.decode_to_codeword(r)
            # Either a codeword or the received word.
            assert not np.any(code.check(v_hat)) or np.array_equal(v_hat, r)


@pytest.mark.parametrize(
    "mu, delta",
    [(2, 3), (3, 4), (3, 5), (4, 7), (4, 10), (6, 11)],
)
def test_forney_error_values(mu, delta, rng):
    code = komm.ReedSolomonCode(mu, delta)
    field, alpha = code.field, int(code.alpha)
    n, t = 2**mu - 1, (delta - 1) // 2
    points = power(field, alpha, range(1, delta))
    inverses = power(field, alpha, -np.arange(n))
    for w in range(1, t + 1):
        for _ in range(10):
            e = np.zeros(n, dtype=int)
            e_loc = np.sort(rng.choice(n, w, replace=False))
            e[e_loc] = rng.integers(1, 2**mu, w)
            syndrome = horner(field, e, points)
            sigma = berlekamp_algorithm(field, syndrome)
            values = forney_algorithm(field, syndrome, sigma, inverses[e_loc])
            assert np.array_equal(values, e[e_loc])
