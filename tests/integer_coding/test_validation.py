import numpy as np
import pytest

import komm


def test_unary_encode_rejects_negative():
    code = komm.UnaryCode()
    for message in ([-1], [-3], [0, 1, -2]):
        with pytest.raises(ValueError, match="invalid entries"):
            code.encode(message)


def test_unary_encode_accepts_zero():
    code = komm.UnaryCode()
    np.testing.assert_equal(code.encode([0]), [0])


def test_fibonacci_encode_rejects_nonpositive():
    code = komm.FibonacciCode()
    for message in ([0], [-1], [1, 0, 2]):
        with pytest.raises(ValueError, match="invalid entries"):
            code.encode(message)


@pytest.mark.parametrize(
    "code, stream",
    [
        (komm.UnaryCode(), [1, 1, 1]),
        (komm.FibonacciCode(), [1, 0, 1, 0]),
        (komm.FibonacciCode(), [0, 0, 0]),
    ],
)
def test_decode_rejects_incomplete_codeword(code, stream):
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode(stream)


@pytest.mark.parametrize(
    "code, stream",
    [
        (komm.UnaryCode(), [1, 2, 0]),
        (komm.UnaryCode(), [0, 5]),
        (komm.FibonacciCode(), [1, 1, 2]),
    ],
)
def test_decode_rejects_non_binary_bits(code, stream):
    with pytest.raises(ValueError, match="invalid entries"):
        code.decode(stream)


def test_valid_roundtrips_unaffected():
    unary = komm.UnaryCode()
    message = list(range(0, 50))  # Unary domain: non-negative integers.
    np.testing.assert_equal(unary.decode(unary.encode(message)), message)
    fibonacci = komm.FibonacciCode()
    message = list(range(1, 50))  # Fibonacci domain: positive integers.
    np.testing.assert_equal(fibonacci.decode(fibonacci.encode(message)), message)
