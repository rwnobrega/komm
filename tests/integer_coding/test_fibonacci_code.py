from math import floor, log, sqrt

import numpy as np
import pytest

import komm


def test_fibonacci_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
    code = komm.FibonacciCode()
    np.testing.assert_equal(code.encode(message), encoded)
    np.testing.assert_equal(code.decode(encoded), message)


@pytest.mark.parametrize("n", range(1, 200))
def test_fibonacci_code_length(n):
    code = komm.FibonacciCode()
    encoded = code.encode([n])
    assert len(encoded) == floor(log(sqrt(5) * (n + 0.5), (1 + sqrt(5)) / 2))


def test_fibonacci_roundtrip():
    code = komm.FibonacciCode()
    message = list(range(1, 50))  # Fibonacci domain: positive integers.
    np.testing.assert_equal(code.decode(code.encode(message)), message)


@pytest.mark.parametrize("message", [[0], [-1], [1, 0, 2]])
def test_fibonacci_encode_rejects_nonpositive(message):
    code = komm.FibonacciCode()
    with pytest.raises(ValueError, match="invalid entries"):
        code.encode(message)


@pytest.mark.parametrize("stream", [[1, 1, 2]])
def test_fobonacci_decode_rejects_non_binary_bits(stream):
    code = komm.FibonacciCode()
    with pytest.raises(ValueError, match="invalid entries"):
        code.decode(stream)


@pytest.mark.parametrize("stream", [[1, 0, 1, 0], [0, 0, 0]])
def test_fibonacci_decode_rejects_incomplete_codeword(stream):
    code = komm.FibonacciCode()
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode(stream)
