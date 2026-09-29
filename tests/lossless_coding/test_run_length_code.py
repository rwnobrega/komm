import numpy as np
import pytest

import komm


def test_run_length_wikipedia():
    # [https://en.wikipedia.org/wiki/Run-length_encoding]
    code = komm.RunLengthCode(source_cardinality=2, max_run_length=32)
    alphabet = "WB"
    source = [
        alphabet.index(x)
        for x in "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW"
    ]
    runs = [(12, "W"), (1, "B"), (12, "W"), (3, "B"), (24, "W"), (1, "B"), (14, "W")]
    tokens = [(alphabet.index(x), length) for length, x in runs]
    assert code.source_to_tokens(source) == tokens
    np.testing.assert_equal(code.tokens_to_source(tokens), source)


@pytest.mark.parametrize(
    "source, tokens",
    [
        ([], []),
        ([2], [(2, 1)]),
        ([1] * 8, [(1, 4), (1, 4)]),
        ([1] * 10, [(1, 4), (1, 4), (1, 2)]),
        ([0, 1, 2, 1, 0], [(0, 1), (1, 1), (2, 1), (1, 1), (0, 1)]),
    ],
)
def test_run_length_special_input(source, tokens):
    code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
    assert code.source_to_tokens(source) == tokens
    np.testing.assert_equal(code.tokens_to_source(tokens), source)


@pytest.mark.parametrize("source_cardinality", [2, 3, 256])
@pytest.mark.parametrize("max_run_length", [1, 4, 7, 100])
def test_run_length_round_trip(source_cardinality, max_run_length, rng):
    code = komm.RunLengthCode(source_cardinality, max_run_length)
    for _ in range(10):
        source = np.repeat(
            rng.integers(0, source_cardinality, 50),
            rng.integers(1, 3 * max_run_length, 50),
        )
        tokens = code.source_to_tokens(source)
        symbols, lengths = np.array(tokens).T
        assert np.all((lengths >= 1) & (lengths <= max_run_length))
        # A run is only split when the earlier token is full
        same = symbols[1:] == symbols[:-1]
        assert np.all(lengths[:-1][same] == max_run_length)
        np.testing.assert_equal(code.tokens_to_source(tokens), source)


def test_run_length_narrow_dtype():
    code = komm.RunLengthCode(source_cardinality=256, max_run_length=100)
    source = np.array([255, 255, 0, 0, 0, 255], dtype=np.uint8)
    tokens = code.source_to_tokens(source)
    assert tokens == [(255, 2), (0, 3), (255, 1)]
    assert all(isinstance(x, int) for token in tokens for x in token)


def test_run_length_invalid_input():
    code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
    with pytest.raises(ValueError, match="invalid entries"):
        code.source_to_tokens([0, 1, 3])
    with pytest.raises(ValueError, match="invalid entries"):
        code.source_to_tokens([-1, 1, 2])


def test_run_length_invalid_construction():
    with pytest.raises(ValueError, match="'source_cardinality' must be at least 2"):
        komm.RunLengthCode(source_cardinality=1, max_run_length=4)
    with pytest.raises(ValueError, match="'max_run_length' must be at least 1"):
        komm.RunLengthCode(source_cardinality=2, max_run_length=0)
    with pytest.raises(ValueError, match="'target_cardinality' must be at least 2"):
        komm.RunLengthCode(source_cardinality=2, max_run_length=4, target_cardinality=1)
