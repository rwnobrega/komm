import numpy as np
import pytest

import komm
from komm._algebra.FiniteBifield import find_roots
from komm._error_control_decoders.BerlekampDecoder import berlekamp_algorithm


def test_berlekamp_lin_costello():
    # [LC04, Example 6.5]
    code = komm.BCHCode(mu=4, delta=7)
    decoder = komm.BerlekampDecoder(code)
    field = code.field
    alpha = code.alpha
    r = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    r_poly = komm.BinaryPolynomial.from_coefficients(r)
    syndrome = code.bch_syndrome(r_poly)
    assert syndrome == [field.one, field.one, alpha**10, field.one, alpha**10, alpha**5]
    sigma = berlekamp_algorithm(code, syndrome)
    assert sigma == [field.one, field.one, field.zero, alpha**5]
    roots = set(find_roots(field, sigma))
    assert roots == {alpha**3, alpha**10, alpha**12}
    inv_roots = {root.inverse() for root in roots}
    assert inv_roots == {alpha**12, alpha**5, alpha**3}
    e_loc = {root.logarithm(alpha) for root in inv_roots}
    assert e_loc == {3, 5, 12}
    u_hat = decoder(r)
    assert np.array_equal(u_hat, [0, 0, 0, 0, 0])


@pytest.mark.parametrize("mu, deltas", [(2, [3]), (3, [3, 7]), (4, [3, 5, 7, 15])])
def test_berlekamp_error_correcting_capability(mu, deltas):
    for delta in deltas:
        code = komm.BCHCode(mu, delta)
        decoder = komm.BerlekampDecoder(code)
        for w in range((delta - 1) // 2 + 1):
            for _ in range(10):
                r = np.zeros(code.length, dtype=int)
                error_locations = np.random.choice(code.length, w, replace=False)
                r[error_locations] ^= 1
                print(r)
                assert np.array_equal(decoder(r), np.zeros(code.dimension, dtype=int))
