import pytest

from komm._lossless_coding.util import num_digits


@pytest.mark.parametrize("base", [2, 3, 8, 10, 256])
def test_num_digits(base):
    assert num_digits(0, base) == 0
    assert num_digits(1, base) == 0
    for k in range(2, 70):
        power = base**k
        assert num_digits(power - 1, base) == k
        assert num_digits(power, base) == k
        assert num_digits(power + 1, base) == k + 1


def test_num_digits_float_rounding():
    # ceil(log(size, base)) gives 30, 8, and 4 here
    assert num_digits(2**29, 2) == 29
    assert num_digits(8**7, 8) == 7
    assert num_digits(125, 5) == 3
