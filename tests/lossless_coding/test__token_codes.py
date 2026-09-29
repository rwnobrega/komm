import numpy as np
import pytest

import komm
import komm.abc

LZ_KWARGS = dict(search_size=2**12, lookahead_size=16, source_cardinality=256)


@pytest.fixture(
    params=[
        komm.LempelZiv77Code(**LZ_KWARGS),
        komm.LempelZivSSCode(**LZ_KWARGS),
        komm.LempelZiv78Code(source_cardinality=256),
        komm.RunLengthCode(source_cardinality=256, max_run_length=16),
    ],
    ids=lambda code: type(code).__name__,
)
def code(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def long_source(rng):
    # Long enough for the LZ78 dictionary to outgrow 8-bit pointers.
    return np.repeat(rng.integers(0, 256, 2048), rng.integers(1, 5, 2048))


def test_token_codes_instance(code: komm.abc.TokenCode):
    assert isinstance(code, komm.abc.TokenCode)


def test_token_codes_empty(code: komm.abc.TokenCode):
    assert code.source_to_tokens([]) == []
    np.testing.assert_equal(code.tokens_to_source([]), [])
    np.testing.assert_equal(code.encode([]), [])
    np.testing.assert_equal(code.decode([]), [])


def test_token_codes_composition(code: komm.abc.TokenCode, long_source):
    tokens = code.source_to_tokens(long_source)
    target = code.tokens_to_target(tokens)
    np.testing.assert_equal(code.encode(long_source), target)
    assert code.target_to_tokens(target) == tokens
    np.testing.assert_equal(code.tokens_to_source(tokens), long_source)
    np.testing.assert_equal(code.decode(target), long_source)


def test_token_codes_narrow_dtype(code: komm.abc.TokenCode, long_source):
    # Bytes and unpacked bits arrive as uint8
    source = long_source.astype(np.uint8)
    compressed = code.encode(source)
    for dtype in [np.uint8, np.int64]:
        np.testing.assert_equal(code.decode(compressed.astype(dtype)), source)


def test_token_codes_invalid_input(code: komm.abc.TokenCode):
    with pytest.raises(ValueError, match="invalid entries"):
        code.encode([0, 256])
    with pytest.raises(ValueError, match="invalid entries"):
        code.encode([-1, 0])
