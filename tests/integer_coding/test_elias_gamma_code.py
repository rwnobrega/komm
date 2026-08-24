import pytest

import komm


def test_elias_gamma_basic():
    message = [1, 2, 3, 4, 5]
    encoded = [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
    code = komm.EliasGammaCode()
    assert list(code.encode(message)) == encoded
    assert list(code.decode(encoded)) == message


def test_elias_gamma_mackay():
    # [MacK03, Table 7.2]
    code = komm.EliasGammaCode()
    assert code.encode_single(45) == [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]


@pytest.mark.parametrize("n", range(1, 200))
def test_elias_gamma_code_length(n):
    code = komm.EliasGammaCode()
    assert code.length(n) == 2 * n.bit_length() - 1


def test_elias_gamma_incomplete_codeword():
    code = komm.EliasGammaCode()
    with pytest.raises(ValueError, match="incomplete codeword"):
        code.decode_single(iter([0, 0, 1, 0]))


def test_elias_gamma_invalid_tail_bit():
    code = komm.EliasGammaCode()
    with pytest.raises(ValueError, match="invalid bit"):
        code.decode_single(iter([0, 0, 1, 0, 7]))
