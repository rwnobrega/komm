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


def test_unary_stop_bit_zero():
    message = [1, 2, 3, 4, 5]
    encoded = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    code = komm.UnaryCode(stop_bit=0)
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


@pytest.mark.parametrize("stop_bit", [-1, 2])
def test_unary_invalid_stop_bit(stop_bit):
    with pytest.raises(ValueError, match=r"'stop_bit' must be in \[0:2\)"):
        komm.UnaryCode(stop_bit=stop_bit)


def test_unary_repr():
    assert repr(komm.UnaryCode()) == "UnaryCode(stop_bit=1)"
    assert repr(komm.UnaryCode(stop_bit=0)) == "UnaryCode(stop_bit=0)"
