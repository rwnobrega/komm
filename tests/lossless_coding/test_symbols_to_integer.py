import numpy as np

from komm._lossless_coding.util import symbols_to_integer


def test_symbols_to_integer_numpy_input():
    # Accumulate in Python integers, not NumPy scalars
    word = [1] * 100
    expected = 2**100 - 1
    assert symbols_to_integer(word, base=2) == expected
    for dtype in [np.uint8, np.int8, np.uint16, np.int32, np.int64]:
        got = symbols_to_integer(np.array(word, dtype=dtype), base=2)
        assert got == expected
        assert isinstance(got, int) and not isinstance(got, np.generic)
    assert symbols_to_integer(np.array([65, 66], dtype=np.uint8), base=256) == 16706
