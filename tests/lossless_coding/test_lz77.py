from itertools import product
from math import ceil

import numpy as np
import pytest

import komm


def test_lz77_empty_and_single():
    code = komm.LempelZiv77Code(
        window_size=16,
        lookahead_size=4,
        source_cardinality=2,
        target_cardinality=2,
    )
    assert code.encode([]).tolist() == []
    assert code.decode([]).tolist() == []
    for symbol in [0, 1]:
        msg = [symbol]
        np.testing.assert_equal(code.decode(code.encode(msg)), msg)


@pytest.mark.parametrize("source_cardinality", [2, 4, 8])
def test_lz77_roundtrip_random(source_cardinality):
    code = komm.LempelZiv77Code(
        window_size=32,
        lookahead_size=8,
        source_cardinality=source_cardinality,
    )

    msg = np.random.randint(0, source_cardinality, size=100).tolist()
    np.testing.assert_equal(code.decode(code.encode(msg)), msg)


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


@pytest.mark.parametrize(
    "alphabet, message, triples, window_size, lookahead_size, len_compressed",
    [
        # [Abrantes, p. 26]
        (
            "ABC",
            "AAAABABCCAABACCAAAABC",
            [
                (0, 0, 0),
                (1, 3, 1),
                (2, 2, 2),
                (1, 1, 0),
                (7, 3, 2),
                (6, 3, 0),
                (8, 2, 2),
            ],
            12,
            4,
            (ceil(np.log2(8 + 1)) + ceil(np.log2(4 + 1)) + ceil(np.log2(len("ABC"))))
            * 7,
        ),
        # [https://blog.coderspirit.xyz/blog/2023/06/04/exploring-the-lz77-algorithm]
        (
            "abdkr",
            "abadakadabra",
            [
                (0, 0, 0),
                (0, 0, 1),
                (2, 1, 2),
                (4, 1, 3),
                (4, 3, 1),
                (0, 0, 4),
                (0, 0, 0),
            ],
            8,
            4,
            (ceil(np.log2(8 + 1)) + ceil(np.log2(4 + 1)) + ceil(np.log2(len("abdkr"))))
            * 7,
        ),
        # [Sayood, p. 122]
        (
            "abcdr",
            "cabracadabrarrarrad",
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
            13,
            6,
            (ceil(np.log2(13 + 1)) + ceil(np.log2(6 + 1)) + ceil(np.log2(len("abcdr"))))
            * 8,
        ),
    ],
)
def test_lz77_examples(
    alphabet, message, triples, window_size, lookahead_size, len_compressed
):
    code = komm.LempelZiv77Code(
        window_size=window_size,
        lookahead_size=lookahead_size,
        source_cardinality=len(alphabet),
        target_cardinality=2,
    )

    msg_indices = [alphabet.index(char) for char in message]

    print(code.source_to_triples(msg_indices))
    np.testing.assert_equal(code.source_to_triples(msg_indices), triples)
    compressed = code.encode(msg_indices)

    assert len(compressed) == len_compressed
    np.testing.assert_equal(code.decode(compressed), msg_indices)
