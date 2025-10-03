import numpy as np
import pytest

import komm


def test_lz77_original_paper():
    code = komm.LempelZiv77Code(
        window_size=18,
        lookahead_size=9,
        source_cardinality=3,
        target_cardinality=3,
    )
    source = [0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2, 0, 2, 1, 0, 2, 1, 2, 0, 0]
    tokens = [(8, 2, 1), (7, 3, 2), (6, 7, 2), (2, 8, 0)]
    target = [2, 2, 0, 2, 1, 2, 1, 1, 0, 2, 2, 0, 2, 1, 2, 0, 2, 2, 2, 0]
    np.testing.assert_equal(code.source_to_tokens(source), tokens)
    np.testing.assert_equal(code.tokens_to_target(tokens), target)
    np.testing.assert_equal(code.tokens_to_source(tokens), source)
    np.testing.assert_equal(code.target_to_tokens(target), tokens)
    np.testing.assert_equal(code.decode(code.encode(source)), source)


def test_lz77_abrantes():
    # [Abrantes, p. 16]
    code = komm.LempelZiv77Code(
        window_size=12,
        lookahead_size=4,
        source_cardinality=3,
        search_buffer=[255] * 8,
    )
    alphabet = "ABC"
    source = [alphabet.index(x) for x in "AAAABABCCAABACCAAAABC"]
    expected = [
        (1, 0, "A"),
        (1, 3, "B"),
        (2, 2, "C"),
        (1, 1, "A"),
        (7, 3, "C"),
        (6, 3, "A"),
        (8, 2, "C"),
    ]
    tokens = []
    for offset, length, symbol in expected:
        tokens.append((8 - offset, length, alphabet.index(symbol)))
    np.testing.assert_equal(code.source_to_tokens(source), tokens)
    np.testing.assert_equal(code.tokens_to_source(tokens), source)
    np.testing.assert_equal(code.decode(code.encode(source)), source)


def test_lz77_sayood():
    # [Sayood, p. 122]
    code = komm.LempelZiv77Code(
        window_size=13,
        lookahead_size=6,
        source_cardinality=256,
        search_buffer=[ord(x) for x in "cabraca"],
    )
    source = [ord(x) for x in "dabrarrarrad"]
    expected = [(1, 0, "d"), (7, 4, "r"), (3, 5, "d")]
    tokens = []
    for offset, length, symbol in expected:
        tokens.append((7 - offset, length, ord(symbol)))
    np.testing.assert_equal(code.source_to_tokens(source), tokens)
    np.testing.assert_equal(code.tokens_to_source(tokens), source)
    np.testing.assert_equal(code.decode(code.encode(source)), source)


def test_lz77_wikipedia():
    # [https://pt.wikipedia.org/wiki/LZ77]
    code = komm.LempelZiv77Code(12, 4, 256)
    expected = [
        (1, 0, "A"),
        (1, 0, "_"),
        (2, 1, "S"),
        (4, 2, "D"),
        (3, 2, "C"),
        (8, 3, "."),
    ]
    source = [ord(x) for x in "A_ASA_DA_CASA."]
    tokens = []
    for offset, length, symbol in expected:
        tokens.append((8 - offset, length, ord(symbol)))
    target = code.encode(source)
    # Wikipedia considers 4 + 3 + 8 = 15 bits instead of 3 + 2 + 8 = 13 bits per codeword
    assert target.size == 13 * len(expected)
    np.testing.assert_equal(code.source_to_tokens(source), tokens)
    np.testing.assert_equal(code.tokens_to_source(tokens), source)
    np.testing.assert_equal(code.decode(target), source)


@pytest.mark.parametrize(
    "source",
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 1, 2, 0, 0],
    ],
)
def test_lz77_special_input(source):
    code = komm.LempelZiv77Code(7, 3, 3)
    tokens = code.source_to_tokens(source)
    target = code.tokens_to_target(tokens)
    np.testing.assert_equal(code.source_to_tokens(source), tokens)
    np.testing.assert_equal(code.tokens_to_target(tokens), target)
    np.testing.assert_equal(code.tokens_to_source(tokens), source)
    np.testing.assert_equal(code.target_to_tokens(target), tokens)
    np.testing.assert_equal(code.decode(code.encode(source)), source)


@pytest.mark.parametrize("parameters", [(12, 8), (15, 8), (16, 8), (20, 17), (31, 17)])
@pytest.mark.parametrize("source_cardinality", [2, 4, 8])
@pytest.mark.parametrize("target_cardinality", [2, 4, 8])
def test_lz77_general(parameters, source_cardinality, target_cardinality):
    window_size, lookahead_size = parameters
    code = komm.LempelZiv77Code(
        window_size, lookahead_size, source_cardinality, target_cardinality
    )

    # Check code parameters
    assert code.window_size == window_size
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
def test_lz77_zeros(k):
    code = komm.LempelZiv77Code(window_size=32, lookahead_size=8, source_cardinality=2)
    source = np.zeros(k, dtype=int)
    np.testing.assert_equal(code.decode(code.encode(source)), source)
