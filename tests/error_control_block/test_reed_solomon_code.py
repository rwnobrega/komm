import numpy as np
import pytest

import komm
from komm._algebra.bifield import horner, power


def test_reed_solomon_lc_example_7_1():
    # [LC04, Example 7.1]
    code = komm.ReedSolomonCode(mu=6, delta=7)
    assert (code.length, code.dimension, code.redundancy) == (6 * 63, 6 * 57, 6 * 6)
    # Message a(X) = 1 gives v(X) = g(X).
    u = np.zeros(code.dimension, dtype=int)
    u[0] = 1
    v = komm.bits_to_int(code.encode(u), width=6)
    alpha = int(code.alpha)
    g = power(code.field, alpha, [21, 10, 55, 43, 48, 59, 0])
    np.testing.assert_equal(v[:7], g)
    np.testing.assert_equal(v[7:], 0)


@pytest.mark.parametrize(
    "mu, delta",
    [(2, 2), (2, 3), (3, 4), (4, 5), (4, 14), (6, 11)],
)
def test_reed_solomon_codewords(mu, delta):
    code = komm.ReedSolomonCode(mu=mu, delta=delta)
    n, k = 2**mu - 1, 2**mu - delta
    assert (code.length, code.dimension) == (mu * n, mu * k)
    u = np.random.randint(0, 2, (100, code.dimension))
    v = komm.bits_to_int(code.encode(u), width=mu)
    # Roots are α, α^2, ..., α^(δ - 1).
    points = power(code.field, int(code.alpha), np.arange(1, delta))
    assert not horner(code.field, v, points).any()
    # Message symbols are on the right.
    np.testing.assert_equal(v[:, -k:], komm.bits_to_int(u, width=mu))


@pytest.mark.parametrize(
    "mu, delta",
    [(2, 2), (2, 3), (3, 4), (3, 5), (4, 13)],
)
def test_reed_solomon_symbol_distance(mu, delta):
    code = komm.ReedSolomonCode(mu=mu, delta=delta)
    v = komm.bits_to_int(code.codewords()[1:], width=mu)
    assert np.count_nonzero(v, axis=1).min() == delta
    assert code.minimum_distance() >= delta


@pytest.mark.parametrize(
    "mu, delta",
    [(1, 2), (3, 1), (3, 8)],
)
def test_reed_solomon_invalid_parameters(mu, delta):
    with pytest.raises(ValueError):
        komm.ReedSolomonCode(mu=mu, delta=delta)
