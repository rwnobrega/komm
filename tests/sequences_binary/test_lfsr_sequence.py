from re import escape

import numpy as np
import pytest

import komm


def test_lfsr_sequence():
    lfsr = komm.LFSRSequence(
        feedback_polynomial=komm.BinaryPolynomial.from_exponents([5, 2, 0])
    )
    np.testing.assert_equal(
        lfsr.bit_sequence,
        # fmt: off
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        # fmt: on
    )
    lfsr = komm.LFSRSequence(feedback_polynomial=0b10000001001)
    np.testing.assert_equal(
        lfsr.bit_sequence[:200],
        komm.to_binary(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
    )


@pytest.mark.parametrize(
    "args, message",
    [
        ((0b0,), "'feedback_polynomial' must have degree at least 1 (got 0b0)"),
        ((0b1,), "'feedback_polynomial' must have degree at least 1 (got 0b1)"),
        ((0b10,), "'feedback_polynomial' must have constant term 1 (got 0b10)"),
        ((0b100100,), "'feedback_polynomial' must have constant term 1"),
        ((0b100101, 0b0), "'start_state_polynomial' must be nonzero"),
        ((0b100101, 0b100000), "'start_state_polynomial' must have degree less than 5"),
    ],
)
def test_lfsr_sequence_invalid(args, message):
    with pytest.raises(ValueError, match=escape(message)):
        komm.LFSRSequence(*args)


@pytest.mark.parametrize("degree", range(2, 13))
def test_lfsr_mls(degree):
    lfsr = komm.LFSRSequence.maximum_length_sequence(degree)
    assert degree == lfsr.feedback_polynomial.degree
    assert lfsr.length == 2**lfsr.feedback_polynomial.degree - 1
    cyclic_autocorrelation = np.full(lfsr.length, fill_value=-1, dtype=int)
    cyclic_autocorrelation[0] = lfsr.length
    np.testing.assert_equal(lfsr.cyclic_autocorrelation(), cyclic_autocorrelation)


@pytest.mark.parametrize("degree", [-10, 0, 25])
def test_lfsr_mls_invalid(degree):
    with pytest.raises(ValueError, match=r"only degrees in the range \[1 : 24\]"):
        komm.LFSRSequence.maximum_length_sequence(degree)


def test_lfsr_mls_degree_not_integer():
    with pytest.raises(TypeError, match=r"'degree' must be an integer \(got float\)"):
        komm.LFSRSequence.maximum_length_sequence(4.0)  # type: ignore
