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
def test_unary_decode_rejects_non_binary(stream):
    code = komm.UnaryCode()
    with pytest.raises(ValueError, match="invalid bit"):
        list(code.decode(stream))


def test_unary_incomplete_codeword():
    code = komm.UnaryCode()
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([0, 0, 0]))


def test_unary_invalid_tail_bit():
    code = komm.UnaryCode()
    with pytest.raises(ValueError, match="invalid bit"):
        code.decode_single(iter([0, 0, 0, 7]))
