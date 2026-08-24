import pytest

import komm


def test_unary_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    code = komm.UnaryCode()
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


@pytest.mark.parametrize("n", range(1, 200))
def test_unary_code_length(n):
    code = komm.UnaryCode()
    assert code.length(n) == n


@pytest.mark.parametrize("stream", [[1, 2, 0], [0, 5]])
def test_unary_decode_rejects_non_binary_bits(stream):
    code = komm.UnaryCode()
    with pytest.raises(ValueError, match="invalid bit"):
        list(code.decode(stream))


@pytest.mark.parametrize("stream", [[0, 0, 0]])
def test_unary_decode_rejects_incomplete_codeword(stream):
    code = komm.UnaryCode()
    with pytest.raises(ValueError, match="incomplete codeword"):
        list(code.decode(stream))
