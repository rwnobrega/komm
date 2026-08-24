import pytest

import komm


def test_elias_delta_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    code = komm.EliasDeltaCode()
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


def test_elias_delta_mackay():
    # [MacK03, Table 7.2]
    code = komm.EliasDeltaCode()
    assert code.encode_single(45) == [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]


@pytest.mark.parametrize("n", range(1, 200))
def test_elias_delta_code_length(n):
    code = komm.EliasDeltaCode()
    num_bits = n.bit_length()
    assert code.length(n) == 2 * num_bits.bit_length() - 1 + num_bits - 1


def test_elias_delta_incomplete_codeword():
    code = komm.EliasDeltaCode()
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([0, 1, 1, 0]))


def test_elias_delta_invalid_tail_bit():
    code = komm.EliasDeltaCode()
    with pytest.raises(ValueError, match="invalid bit"):
        code.decode_single(iter([0, 1, 1, 0, 7]))
