import pytest

from komm._lossless_coding.util import ceil_log


@pytest.mark.parametrize("base", [2, 3, 8, 10, 256])
def test_ceil_log(base):
    assert ceil_log(0, base) == 0
    assert ceil_log(1, base) == 0
    for k in range(2, 70):
        power = base**k
        assert ceil_log(power - 1, base) == k
        assert ceil_log(power, base) == k
        assert ceil_log(power + 1, base) == k + 1


def test_ceil_log_float_rounding():
    # ceil(log(size, base)) gives 30, 8, and 4 here
    assert ceil_log(2**29, 2) == 29
    assert ceil_log(8**7, 8) == 7
    assert ceil_log(125, 5) == 3
