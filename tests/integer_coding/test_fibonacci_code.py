from math import floor, log, sqrt

import pytest

import komm


def test_fibonacci_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
    code = komm.FibonacciCode()
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


@pytest.mark.parametrize("n", range(1, 200))
def test_fibonacci_code_length(n):
    code = komm.FibonacciCode()
    assert code.length(n) == floor(log(sqrt(5) * (n + 0.5), (1 + sqrt(5)) / 2))


@pytest.mark.parametrize("stream", [[1, 0, 1, 2]])
def test_fibonacci_decode_rejects_non_binary_bits(stream):
    code = komm.FibonacciCode()
    with pytest.raises(ValueError, match="invalid bit"):
        list(code.decode(stream))


@pytest.mark.parametrize("stream", [[1, 0, 1, 0], [0, 0, 0]])
def test_fibonacci_decode_rejects_incomplete_codeword(stream):
    code = komm.FibonacciCode()
    with pytest.raises(ValueError, match="incomplete codeword"):
        list(code.decode(stream))
