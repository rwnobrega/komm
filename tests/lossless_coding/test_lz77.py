from itertools import product
from math import ceil, log2

import numpy as np
import pytest

import komm


# Helper function for target_cardinality = 2 only
def len_compressed(A, W, L, num_triples):
    return (ceil(log2(W + 1)) + ceil(log2(L + 1)) + ceil(log2(A))) * num_triples


@pytest.mark.parametrize(
    "alphabet, message, parameters, triples",
    [
        # [Abrantes, p. 26]
        (
            "ABC",
            "AAAABABCCAABACCAAAABC",
            (12, 4),
            [
                (0, 0, 0),
                (1, 3, 1),
                (2, 2, 2),
                (1, 1, 0),
                (7, 3, 2),
                (6, 3, 0),
                (8, 2, 2),
            ],
        ),
        # [https://blog.coderspirit.xyz/blog/2023/06/04/exploring-the-lz77-algorithm]
        (
            "abdkr",
            "abadakadabra",
            (8, 4),
            [
                (0, 0, 0),
                (0, 0, 1),
                (2, 1, 2),
                (4, 1, 3),
                (4, 3, 1),
                (0, 0, 4),
                (0, 0, 0),
            ],
        ),
        # [Sayood, p. 122]
        (
            "abcdr",
            "cabracadabrarrarrad",
            (13, 6),
            [
                (0, 0, 2),
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 4),
                (3, 1, 2),
                (5, 1, 3),
                (7, 4, 4),
                (3, 5, 3),
            ],
        ),
    ],
)
def test_lz77_literature(alphabet, message, parameters, triples):
    window_size, lookahead_size = parameters
    code = komm.LempelZiv77Code(
        window_size=window_size,
        lookahead_size=lookahead_size,
        source_cardinality=len(alphabet),
        target_cardinality=2,
    )
    A, W, L = len(alphabet), window_size, lookahead_size
    message = [alphabet.index(char) for char in message]
    np.testing.assert_equal(code.source_to_triples(message), triples)
    compressed = code.encode(message)
    assert len(compressed) == len_compressed(A, W, L, len(triples))
    np.testing.assert_equal(code.decode(compressed), message)


@pytest.mark.parametrize("window_size", [16, 20, 31])
@pytest.mark.parametrize("lookahead_size", [8, 12, 17])
@pytest.mark.parametrize("source_cardinality", [2, 4, 8])
@pytest.mark.parametrize("target_cardinality", [2, 4, 8])
def test_lz77_general(
    window_size,
    lookahead_size,
    source_cardinality,
    target_cardinality,
):
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
    message = np.random.randint(0, source_cardinality, 1000)
    np.testing.assert_equal(code.decode(code.encode(message)), message)


@pytest.mark.parametrize("k", range(2, 6))
def test_lz77_zero_runs(k):
    code = komm.LempelZiv77Code(window_size=32, lookahead_size=8, source_cardinality=2)
    msg = []
    for r in range(1, k + 1):
        msg.extend([0] * r)
    compressed = code.encode(msg)
    np.testing.assert_equal(code.decode(compressed), msg)


@pytest.mark.parametrize("k", range(2, 5))
def test_lz77_worst_case(k):
    code = komm.LempelZiv77Code(window_size=64, lookahead_size=16, source_cardinality=2)
    msg = []
    for r in range(1, k + 1):
        for bits in product([0, 1], repeat=r):
            msg.extend(bits)
    compressed = code.encode(msg)
    np.testing.assert_equal(code.decode(compressed), msg)
    assert len(compressed) <= len(msg) * 10  # sanity bound
