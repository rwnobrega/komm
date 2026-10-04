import pytest

import komm


def test_elias_omega_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    code = komm.EliasOmegaCode()
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


@pytest.mark.parametrize(
    "n, codeword",
    [
        (1, "0"),
        (2, "100"),
        (3, "110"),
        (4, "101000"),
        (5, "101010"),
        (6, "101100"),
        (7, "101110"),
        (8, "1110000"),
        (9, "1110010"),
        (10, "1110100"),
        (11, "1110110"),
        (12, "1111000"),
        (13, "1111010"),
        (14, "1111100"),
        (15, "1111110"),
        (16, "10100100000"),
        (31, "10100111110"),
        (32, "101011000000"),
        (45, "101011011010"),
        (63, "101011111110"),
        (64, "1011010000000"),
        (127, "1011011111110"),
        (128, "10111100000000"),
        (255, "10111111111110"),
        (256, "1110001000000000"),
        (365, "1110001011011010"),
        (511, "1110001111111110"),
        (512, "11100110000000000"),
        (719, "11100110110011110"),
        (1023, "11100111111111110"),
        (1024, "111010100000000000"),
        (1025, "111010100000000010"),
    ],
)
def test_elias_omega_mackay(n, codeword):
    # [MacK03, Table 7.5]
    code = komm.EliasOmegaCode()
    assert code.encode_single(n) == [int(bit) for bit in codeword]


@pytest.mark.parametrize(
    "n, codeword",
    [
        (100, "10 110 1100100 0"),
        (1000, "11 1001 1111101000 0"),
        (10000, "11 1101 10011100010000 0"),
        (100000, "10 100 10000 11000011010100000 0"),
        (1000000, "10 100 10011 11110100001001000000 0"),
    ],
)
def test_elias_omega_wikipedia(n, codeword):
    code = komm.EliasOmegaCode()
    assert code.encode_single(n) == [int(bit) for bit in codeword if bit != " "]


def test_elias_omega_wikipedia_googol():
    code = komm.EliasOmegaCode()
    n = 10**100
    codeword = (
        "11 1000 101001100 10010 01001001 10101101 00100101 10010100 11000011 01111100 "
        "11101011 00001011 00100111 10000100 11000100 11001110 00001011 11110011 "
        "10001010 11001110 01000000 10001110 00100001 00011010 01111100 10101010 "
        "10110010 01000011 00001000 10101000 00101110 10001111 00010000 00000000 "
        "00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 "
        "00000000 00000000 00000000 0"
    )
    assert code.encode_single(n) == [int(bit) for bit in codeword if bit != " "]


def test_elias_omega_versus_delta():
    code = komm.EliasOmegaCode()
    delta = komm.EliasDeltaCode()
    assert code.length(2**126) > delta.length(2**126)
    for b in range(128, 1000):
        assert code.length(2 ** (b - 1)) <= delta.length(2 ** (b - 1))


def test_elias_omega_incomplete_codeword():
    code = komm.EliasOmegaCode()
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([1, 0, 1]))


def test_elias_omega_invalid_tail_bit():
    code = komm.EliasOmegaCode()
    with pytest.raises(ValueError, match="invalid bit"):
        code.decode_single(iter([1, 0, 1, 0, 0, 7]))


def test_elias_omega_repr():
    assert repr(komm.EliasOmegaCode()) == "EliasOmegaCode()"
