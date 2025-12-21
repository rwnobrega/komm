from bz2 import compress

import numpy as np
import pytest

import komm


def test_lzss_cover_exercise_13_8():
    # [CT06,  Exercise 13.8 (c), p. 459].
    ws, ls = 4096, 256
    code = komm.LempelZivSSCode(
        search_size=ws - ls,
        lookahead_size=ls,
        source_cardinality=256,
    )
    assert code.break_even == 3


def test_lzss_cover():
    # [CT06, p. 442]
    code = komm.LempelZivSSCode(
        search_size=4,  # [CT06] call it W.
        lookahead_size=32,  # [CT06] considers it arbitrarily long; 32 will suffice here.
        source_cardinality=256,  # Large value to always favor references, as in [CT06].
        search_buffer=[2, 2, 2, 2],  # Forces literals in the beginning, as in [CT06].
    )
    alphabet = "AB"
    source = [alphabet.index(x) for x in "ABBABBABBBAABABA"]
    tokens = [
        (0, 0),
        (0, 1),
        (1, 1, 1),
        (1, 3, 6),
        (1, 4, 2),
        (1, 1, 1),
        (1, 3, 2),
        (1, 2, 2),
    ]
    assert code.break_even == 1
    np.testing.assert_equal(code.source_to_tokens(source), tokens)
    np.testing.assert_equal(code.tokens_to_source(tokens), source)
    np.testing.assert_equal(code.decode(code.encode(source)), source)


@pytest.mark.parametrize(
    "source, compressed_size",
    [
        ("zzzzz", 26),
        ("wxyzwxy", 53),
        ("wxyzwxyzwxy", 53),
    ],
)
def test_lzss_tim_cogan(source, compressed_size):
    # https://tim.cogan.dev/lzss/
    code = komm.LempelZivSSCode(
        search_size=2**12,
        lookahead_size=2**4,
        source_cardinality=256,
    )
    source = [ord(x) for x in source]
    assert code.encode(source).size == compressed_size


@pytest.mark.parametrize("parameters", [(4, 8), (8, 4), (7, 8), (8, 8), (1, 8)])
@pytest.mark.parametrize("source_cardinality", [2, 4, 8])
@pytest.mark.parametrize("target_cardinality", [2, 4, 8])
def test_lzss_general(parameters, source_cardinality, target_cardinality):
    search_size, lookahead_size = parameters
    code = komm.LempelZivSSCode(
        search_size=search_size,
        lookahead_size=lookahead_size,
        source_cardinality=source_cardinality,
        target_cardinality=target_cardinality,
    )

    # Check code parameters
    assert code.window_size == search_size + lookahead_size
    assert code.search_size == search_size
    assert code.lookahead_size == lookahead_size
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality

    # Encode and decode empty input
    np.testing.assert_equal(code.encode([]), [])
    np.testing.assert_equal(code.decode([]), [])

    # Single-symbol message
    for symbol in range(source_cardinality):
        np.testing.assert_equal(code.decode(code.encode([symbol])), [symbol])

    # Random message
    source = np.random.randint(0, source_cardinality, 1000)
    np.testing.assert_equal(code.decode(code.encode(source)), source)


@pytest.mark.parametrize("k", range(100))
def test_lzss_zeros(k):
    code = komm.LempelZivSSCode(search_size=24, lookahead_size=8, source_cardinality=2)
    source = np.zeros(k, dtype=int)
    np.testing.assert_equal(code.decode(code.encode(source)), source)


def test_lzss_invalid_input():
    code = komm.LempelZivSSCode(search_size=24, lookahead_size=8, source_cardinality=27)
    code.encode([0, 10, 26])
    with pytest.raises(ValueError, match="invalid entries"):
        code.encode([0, 10, 27])
    with pytest.raises(ValueError, match="invalid entries"):
        code.encode([-1, 10, 26])


def test_lzss_invalid_construction():
    with pytest.raises(ValueError, match="'search_size' must be at least 1"):
        komm.LempelZivSSCode(
            search_size=0,
            lookahead_size=4,
            source_cardinality=2,
        )

    with pytest.raises(ValueError, match="'lookahead_size' must be at least 1"):
        komm.LempelZivSSCode(
            search_size=4,
            lookahead_size=0,
            source_cardinality=2,
        )

    with pytest.raises(ValueError, match="'source_cardinality' must be at least 2"):
        komm.LempelZivSSCode(
            search_size=4,
            lookahead_size=4,
            source_cardinality=1,
        )

    with pytest.raises(ValueError, match="'target_cardinality' must be at least 2"):
        komm.LempelZivSSCode(
            search_size=4,
            lookahead_size=4,
            source_cardinality=2,
            target_cardinality=1,
        )


def test_lzss_invalid_construction_search_buffer():
    komm.LempelZivSSCode(
        search_size=4,
        lookahead_size=3,
        source_cardinality=2,
        search_buffer=[0, 0, 0, 0],
    )
    with pytest.raises(ValueError, match="length of 'search_buffer' must"):
        komm.LempelZivSSCode(
            search_size=4,
            lookahead_size=3,
            source_cardinality=2,
            search_buffer=[0, 0, 0],
        )
