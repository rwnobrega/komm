from math import ceil, log2

import numpy as np
import pytest

import komm


# Helper function for target_cardinality = 2 only
def len_compressed(dict_size, source_cardinality):
    S = source_cardinality
    return sum(ceil(log2(S + i)) for i in range(dict_size))


ASCII = [chr(n) for n in range(256)]


def test_lzw_wikipedia():
    alphabet = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    code = komm.LempelZivWelchCode(len(alphabet))
    message = [alphabet.index(char) for char in "TOBEORNOTTOBEORTOBEORNOT#"]
    raw = "10100|01111|00010|00101|01111|10010|001110|001111|010100|011011|011101|011111|100100|011110|100000|100010|000000"
    # Convert the pointers from MSB-first to LSB-first.
    compressed = [int(char) for char in "".join(word[::-1] for word in raw.split("|"))]
    np.testing.assert_array_equal(code.encode(message), compressed)


@pytest.mark.parametrize(
    "alphabet, message, dict_size",
    [
        # Prof. Saquib Razak - Lempel–Ziv Compression Techniques - ICS202
        # https://faculty.kfupm.edu.sa/ICS/saquib/ICS202/Unit32_LZW.pdf
        (ASCII, "BABAABAAA", 6),
        (ASCII, "BABAABRRRA", 7),
        (ASCII, "aaabbbbbbaabaaba", 7),
        (ASCII, "CFCFCFCCFCCFC", 6),
        # [Say06, Example 5.4.2]
        (" abow", "wabba wabba wabba wabba woo woo woo", 21),
    ],
)
def test_lzw_literature(alphabet, message, dict_size):
    code = komm.LempelZivWelchCode(len(alphabet))
    message = [alphabet.index(char) for char in message]
    compressed = code.encode(message)
    assert len(compressed) == len_compressed(dict_size, len(alphabet))
    np.testing.assert_array_equal(code.decode(compressed), message)


@pytest.mark.parametrize("source_cardinality", range(2, 21))
@pytest.mark.parametrize("target_cardinality", range(2, 21))
def test_lzw_encode_decode(source_cardinality, target_cardinality):
    code = komm.LempelZivWelchCode(source_cardinality, target_cardinality)

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
