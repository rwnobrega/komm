import pytest

import komm


def test_escape_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    code = komm.EscapeCode(3)
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


def test_escape_mixed_blocks():
    code = komm.EscapeCode(4)
    codewords = "00 01 10 110 11100 11101 11110 111110 11111100"
    for n, codeword in enumerate(codewords.split(), start=1):
        assert code.encode_single(n) == [int(bit) for bit in codeword]


@pytest.mark.parametrize("M", [1, 2, 3, 4, 5])
def test_escape_block_sequences(M):
    code = komm.EscapeCode(M)
    block_code = komm.TruncatedBinaryCode(M + 1)
    sequences = ([M] * q + [r] for q in range(4) for r in range(M))
    for n, values in enumerate(sequences, start=1):
        assert code.encode_single(n) == list(block_code.encode(values))


def test_escape_unary():
    code = komm.EscapeCode(1)
    unary = komm.UnaryCode(stop_bit=0)
    for n in range(1, 100):
        assert code.encode_single(n) == unary.encode_single(n)


@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_escape_fixed_blocks(b):
    M = 2**b - 1
    code = komm.EscapeCode(M)
    for n in range(1, 1000):
        assert code.length(n) == b * ((n - 1) // M + 1)


@pytest.mark.parametrize("M", [0, -1])
def test_escape_invalid_divisor(M):
    with pytest.raises(ValueError, match="'divisor' must be at least 1"):
        komm.EscapeCode(M)


def test_escape_incomplete_codeword():
    code = komm.EscapeCode(3)
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([1, 1, 0]))


def test_escape_invalid_tail_bit():
    code = komm.EscapeCode(3)
    with pytest.raises(ValueError, match="invalid bit"):
        code.decode_single(iter([1, 1, 0, 7]))


def test_escape_repr():
    assert repr(komm.EscapeCode(3)) == "EscapeCode(divisor=3)"
