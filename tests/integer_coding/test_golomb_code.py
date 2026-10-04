import pytest

import komm


def test_golomb_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
    code = komm.GolombCode(3)
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


@pytest.mark.parametrize(
    "M, codewords",
    [
        # fmt: off
        (1, "0 10 110 1110 11110 111110 1111110 11111110 111111110 1111111110 11111111110"),
        (2, "00 01 100 101 1100 1101 11100 11101 111100 111101 1111100"),
        (3, "00 010 011 100 1010 1011 1100 11010 11011 11100 111010"),
        (4, "000 001 010 011 1000 1001 1010 1011 11000 11001 11010"),
        (14,
            "0000 0001 "
            "00100 00101 00110 00111 01000 01001 01010 01011 01100 01101 01110 01111 10000 10001 "
            "100100 100101 100110 100111 101000 101001 101010 101011 101100 101101 101110 101111 110000 110001 "
            "1100100 1100101 1100110 1100111 1101000 1101001 1101010 1101011 1101100 1101101 1101110 1101111 1110000 1110001 "
            "11100100 11100101 11100110 11100111"
        ),
        (16,
            "00000 00001 00010 00011 00100 00101 00110 00111 01000 01001 01010 01011 01100 01101 01110 01111 "
            "100000 100001 100010 100011 100100 100101 100110 100111 101000 101001 101010 101011 101100 101101 101110 101111 "
            "1100000 1100001 1100010 1100011 1100100 1100101 1100110 1100111 1101000 1101001 1101010 1101011 1101100 1101101 1101110 1101111"
        ),
        # fmt: on
    ],
)
def test_golomb_paper_tables(M, codewords):
    # [Gol66, Tables I and II], shifted by one
    code = komm.GolombCode(M)
    for n, codeword in enumerate(codewords.split(), start=1):
        assert code.encode_single(n) == [int(bit) for bit in codeword]


def test_golomb_unary():
    code = komm.GolombCode(1)
    unary = komm.UnaryCode(stop_bit=0)
    for n in range(1, 100):
        assert code.encode_single(n) == unary.encode_single(n)


@pytest.mark.parametrize("M", [0, -1])
def test_golomb_invalid_divisor(M):
    with pytest.raises(ValueError, match="'divisor'"):
        komm.GolombCode(M)


def test_golomb_incomplete_codeword():
    code = komm.GolombCode(3)
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([1, 0, 1]))


def test_golomb_invalid_tail_bit():
    code = komm.GolombCode(3)
    with pytest.raises(ValueError, match="invalid bit"):
        code.decode_single(iter([1, 0, 7]))


def test_golomb_repr():
    assert repr(komm.GolombCode(3)) == "GolombCode(divisor=3)"
