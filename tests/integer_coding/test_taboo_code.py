from itertools import product

import pytest

import komm


def test_taboo_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]
    code = komm.TabooCode(3)
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


def test_taboo_book_table():
    # [SM10, Table 3.32], shifted by two
    code = komm.TabooCode(3)
    codewords = (
        "0100 1000 1100 "
        "010100 011000 011100 100100 101000 101100 110100 111000 111100 "
        "01010100 01011000 01011100"
    )
    for n, codeword in enumerate(codewords.split(), start=2):
        assert code.encode_single(n) == [int(bit) for bit in codeword]


@pytest.mark.parametrize("M", [1, 2, 3, 4, 5])
def test_taboo_digit_sequences(M):
    code = komm.TabooCode(M)
    block_code = komm.TruncatedBinaryCode(M + 1)
    sequences = (s for k in range(4) for s in product(range(1, M + 1), repeat=k))
    for n, digits in enumerate(sequences, start=1):
        assert code.encode_single(n) == list(block_code.encode([*digits, 0]))


def test_taboo_unary():
    code = komm.TabooCode(1)
    unary = komm.UnaryCode(stop_bit=0)
    for n in range(1, 100):
        assert code.encode_single(n) == unary.encode_single(n)


def test_taboo_gamma_lengths():
    code = komm.TabooCode(2)
    gamma = komm.EliasGammaCode()
    for n in range(1, 1000):
        assert code.length(n) == gamma.length(n)


@pytest.mark.parametrize("M", [0, -1])
def test_taboo_invalid_base(M):
    with pytest.raises(ValueError, match="'base' must be a positive integer"):
        komm.TabooCode(M)


def test_taboo_incomplete_codeword():
    code = komm.TabooCode(3)
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([0, 1, 0]))


def test_taboo_invalid_tail_bit():
    code = komm.TabooCode(3)
    with pytest.raises(ValueError, match="invalid bit"):
        code.decode_single(iter([0, 1, 0, 7]))


def test_taboo_repr():
    assert repr(komm.TabooCode(3)) == "TabooCode(base=3)"
