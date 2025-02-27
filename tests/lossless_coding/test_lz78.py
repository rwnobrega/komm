from itertools import product
from math import ceil, log2

import numpy as np
import pytest

import komm


# Helper function for target_cardinality = 2 only
def len_compressed(dict_size, source_cardinality, incomplete=False):
    M = ceil(log2(source_cardinality))
    length = sum(ceil(log2(i)) + M for i in range(1, dict_size + 1))
    return length - M if incomplete else length


ASCII = [chr(n) for n in range(256)]


@pytest.mark.parametrize(
    "alphabet, message, dict_size, incomplete",
    [
        # Prof. Saquib Razak - Lempel–Ziv Compression Techniques - ICS202
        # https://faculty.kfupm.edu.sa/ICS/saquib/ICS202/Unit31_LZ78.pdf
        (ASCII, "ABBCBCABABCAABCAAB", 7, False),
        (ASCII, "BABAABRRRA", 7, True),
        (ASCII, "AAAAAAAAA", 4, True),
        # [Say06, Example 5.4.2]
        (" abow", "wabba wabba wabba wabba woo woo woo", 17, False),
    ],
)
def test_lz78_literature(alphabet, message, dict_size, incomplete):
    code = komm.LempelZiv78Code(len(alphabet))
    message = [alphabet.index(char) for char in message]
    compressed = code.encode(message)
    assert len(compressed) == len_compressed(dict_size, len(alphabet), incomplete)
    np.testing.assert_array_equal(code.decode(compressed), message)


def test_lz78_shor():
    # Prof. Peter Shor: Lempel–Ziv Notes - 18.310C, Spring 2010
    # https://math.mit.edu/~shor/18.310/lempel_ziv_notes.pdf
    code = komm.LempelZiv78Code(2)
    message = [ord(char) - ord("A") for char in "AABABBBABAABABBBABBABB"]
    compressed = [
        int(char)
        # Below we manually converted the pointers from MSB-first to LSB-first.
        for char in ",0 | 1,1 | 01,1 | 00,1 | 010,0 | 101,1 | 001,1 | 110,0 | 1110"
        if char in "01"
    ]
    np.testing.assert_array_equal(code.encode(message), compressed)
    np.testing.assert_array_equal(code.decode(compressed), message)


@pytest.mark.parametrize("source_cardinality", range(2, 21))
@pytest.mark.parametrize("target_cardinality", range(2, 21))
def test_lz78_general(source_cardinality, target_cardinality):
    code = komm.LempelZiv78Code(source_cardinality, target_cardinality)

    # Check code parameters
    assert code.source_cardinality == source_cardinality
    assert code.target_cardinality == target_cardinality

    # Encode and decode empty input
    np.testing.assert_array_equal(code.encode([]), [])
    np.testing.assert_array_equal(code.decode([]), [])

    # Single-symbol message
    for symbol in range(source_cardinality):
        np.testing.assert_array_equal(code.decode(code.encode([symbol])), [symbol])

    # Random message
    message = np.random.randint(0, source_cardinality, 1000)
    np.testing.assert_array_equal(code.decode(code.encode(message)), message)


@pytest.mark.parametrize("k", range(2, 13))
def test_lz78_zero_message(k):
    # All zero message: 0|00|000|0000...
    code = komm.LempelZiv78Code(2)
    message = []
    for r in range(1, k + 1):
        message.extend([0] * r)
    compressed = code.encode(message)
    assert len(message) == k * (k + 1) // 2
    assert len(compressed) == len_compressed(k, 2)
    np.testing.assert_array_equal(code.decode(compressed), message)


@pytest.mark.parametrize("k", range(2, 13))
def test_lz78_worst_case(k):
    # Worst case message: 0|1|00|01|10|11|000|001...
    # Prof. Peter Shor: Lempel–Ziv Notes - 18.310C, Spring 2010
    # https://math.mit.edu/~shor/18.310/lempel_ziv_notes.pdf
    code = komm.LempelZiv78Code(2)
    message = []
    for r in range(1, k + 1):
        for bits in product([0, 1], repeat=r):
            message.extend(bits)
    compressed = code.encode(message)
    assert len(message) == (k - 1) * 2 ** (k + 1) + 2
    assert len(compressed) == len_compressed(2 ** (k + 1) - 2, 2)
    np.testing.assert_array_equal(code.decode(compressed), message)
