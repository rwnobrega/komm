import numpy as np
import pytest

import komm


def test_unary_basic():
    message = [1, 2, 3, 4]
    encoded = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    code = komm.UnaryCode()
    np.testing.assert_equal(code.encode(message), encoded)
    np.testing.assert_equal(code.decode(encoded), message)


@pytest.mark.parametrize("n", range(1, 200))
def test_unary_code_length(n):
    code = komm.UnaryCode()
    encoded = code.encode([n])
    assert len(encoded) == n


def test_unary_roundtrip():
    code = komm.UnaryCode()
    message = list(range(1, 50))  # Unary domain: positive integers.
    np.testing.assert_equal(code.decode(code.encode(message)), message)


@pytest.mark.parametrize("message", [[0], [-1], [1, 0, 2]])
def test_unary_encode_rejects_non_positive(message):
    code = komm.UnaryCode()
    with pytest.raises(ValueError, match="invalid entries"):
        code.encode(message)


@pytest.mark.parametrize("stream", [[1, 2, 0], [0, 5]])
def test_decode_rejects_non_binary_bits(stream):
    code = komm.UnaryCode()
    with pytest.raises(ValueError, match="invalid entries"):
        code.decode(stream)


@pytest.mark.parametrize("stream", [[0, 0, 0]])
def test_unary_decode_rejects_incomplete_codeword(stream):
    code = komm.UnaryCode()
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode(stream)
