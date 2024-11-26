import numpy as np

import komm
from komm._error_control_block.decoders.berlekamp import (
    bch_syndrome,
    berlekamp_algorithm,
    find_roots,
)


def test_bch_syndrome():
    # [LC04, Example 6.4]
    code = komm.BCHCode(mu=4, delta=5)
    alpha = code.field.primitive_element
    r = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    r_poly = komm.BinaryPolynomial.from_coefficients(r)
    assert bch_syndrome(code, r_poly) == [alpha**2, alpha**4, alpha**7, alpha**8]


def test_bch_berlekamp_step_by_step():
    # [LC04, Example 6.5]
    code = komm.BCHCode(mu=4, delta=7)
    field = code.field
    alpha = field.primitive_element
    r = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    r_poly = komm.BinaryPolynomial.from_coefficients(r)
    s_vec = [field.one, field.one, alpha**10, field.one, alpha**10, alpha**5]
    assert s_vec == bch_syndrome(code, r_poly)
    sigma = berlekamp_algorithm(field, code.delta, bch_syndrome(code, r_poly))
    assert sigma == [field.one, field.one, field.zero, alpha**5]
    roots = set(find_roots(field, sigma))
    assert roots == {alpha**3, alpha**10, alpha**12}
    inv_roots = {root.inverse() for root in roots}
    assert inv_roots == {alpha**12, alpha**5, alpha**3}
    e_loc = {root.logarithm(alpha) for root in inv_roots}
    assert e_loc == {3, 5, 12}


def test_bch_berlekamp():
    # [LC04, Example 6.5]
    code = komm.BCHCode(mu=4, delta=7)
    decoder = komm.BlockDecoder(code, method="berlekamp")
    assert np.array_equal(
        decoder([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        [0, 0, 0, 0, 0],
    )


def test_bch_berlekamp_zero_codeword():
    code = komm.BCHCode(mu=4, delta=7)
    decoder = komm.BlockDecoder(code, method="berlekamp")
    assert np.array_equal(decoder([0] * code.length), [0] * code.dimension)
