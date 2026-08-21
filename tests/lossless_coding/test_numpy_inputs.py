import numpy as np
import pytest

import komm
from komm._lossless_coding.util import symbols_to_integer


def test_symbols_to_integer_numpy_input():
    # The accumulator must remain an arbitrary-precision Python integer even
    # when the input symbols are numpy scalars of fixed-width dtype.
    word = [1] * 100
    expected = 2**100 - 1
    assert symbols_to_integer(word, base=2) == expected
    for dtype in [np.uint8, np.int8, np.uint16, np.int32, np.int64]:
        got = symbols_to_integer(np.array(word, dtype=dtype), base=2)
        assert got == expected
        assert isinstance(got, int) and not isinstance(got, np.generic)
    assert symbols_to_integer(np.array([65, 66], dtype=np.uint8), base=256) == 16706


@pytest.fixture
def long_source():
    # Long enough for the LZ78/LZW dictionaries to outgrow 8-bit pointers.
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=8192)


LZ_KWARGS = dict(search_size=2**12, lookahead_size=16, source_cardinality=256)


@pytest.mark.parametrize(
    "code",
    [
        komm.LempelZiv77Code(**LZ_KWARGS),
        komm.LempelZivSSCode(**LZ_KWARGS),
        komm.LempelZiv78Code(source_cardinality=256),
        komm.LempelZivWelchCode(source_cardinality=256),
    ],
    ids=["lz77", "lzss", "lz78", "lzw"],
)
def test_lempel_ziv_narrow_dtype_round_trip(code, long_source):
    # Source symbols stored as bytes and compressed bits unpacked with
    # np.unpackbits both arrive as uint8 arrays; the round trip must not
    # depend on the dtype of the input arrays.
    source = long_source.astype(np.uint8)
    compressed = code.encode(source)
    for dtype in [np.uint8, np.int64]:
        decoded = code.decode(compressed.astype(dtype))
        np.testing.assert_equal(decoded, source)
