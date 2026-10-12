from re import escape

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
        komm.LempelZivWelchCode(source_cardinality=256),
        komm.RunLengthCode(source_cardinality=256, max_run_length=16),
    ],
    ids=lambda code: type(code).__name__,
)
def code(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def long_source(rng):
    # Long enough for LZ78/LZW dictionaries to outgrow 8-bit pointers.
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
    with pytest.raises(ValueError, match="elements of 'source' must be in"):
        code.encode([0, 256])
    with pytest.raises(ValueError, match="elements of 'source' must be in"):
        code.encode([-1, 0])


@pytest.mark.parametrize("side", ["below", "above"])
def test_token_codes_invalid_target(code: komm.abc.TokenCode, side):
    symbol = -1 if side == "below" else code.target_cardinality
    message = f"elements of 'target' must be in [0:{code.target_cardinality})"
    with pytest.raises(ValueError, match=escape(message)):
        code.decode([0, symbol])


def test_token_codes_target_not_integer(code: komm.abc.TokenCode):
    with pytest.raises(TypeError, match="'target' must contain only integers"):
        code.decode([0.0, 1.0])


@pytest.mark.parametrize(
    "method, name",
    [
        ("encode", "source"),
        ("decode", "target"),
    ],
)
def test_token_codes_not_1d(code: komm.abc.TokenCode, method, name):
    message = f"'{name}' must be a 1D-array (got shape (2, 2))"
    with pytest.raises(ValueError, match=escape(message)):
        getattr(code, method)([[0, 1], [1, 0]])


def test_token_codes_truncated_target(code: komm.abc.TokenCode, long_source):
    target = code.encode(long_source)
    with pytest.raises(ValueError, match="input contains an incomplete token"):
        code.decode(target[:-1])


@pytest.mark.parametrize("symbol", [0, 1])
def test_token_codes_extended_target(code: komm.abc.TokenCode, long_source, symbol):
    target = np.append(code.encode(long_source), symbol)
    with pytest.raises(ValueError, match="input contains an incomplete token"):
        code.decode(target)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (komm.LempelZiv77Code, LZ_KWARGS),
        (komm.LempelZivSSCode, LZ_KWARGS),
        (komm.LempelZiv78Code, dict(source_cardinality=256)),
        (komm.LempelZivWelchCode, dict(source_cardinality=256)),
        (komm.RunLengthCode, dict(source_cardinality=256, max_run_length=16)),
    ],
)
def test_token_codes_integer_parameters(cls, kwargs):
    kwargs = kwargs | dict(target_cardinality=2)
    for name, value in kwargs.items():
        code = cls(**kwargs | {name: np.int64(value)})
        assert type(getattr(code, name)) is int
        with pytest.raises(TypeError, match=f"'{name}' must be an integer"):
            cls(**kwargs | {name: float(value)})
