import numpy as np
import pytest

from komm import Lexicode


def test_lexicode_hamming74():
    code = Lexicode(7, 3)
    assert code.length == 7
    assert code.dimension == 4
    assert code.redundancy == 3
    assert code.minimum_distance() == 3
    generator_matrix = [
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 1],
    ]
    np.testing.assert_array_equal(code.generator_matrix, generator_matrix)
    check_matrix = [
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
    ]
    np.testing.assert_array_equal(code.check_matrix, check_matrix)


def test_lexicode_wikipedia():
    # From [https://en.wikipedia.org/wiki/Lexicographic_code].
    wikipedia_lexicodes_parameters = {
        1: [1],
        2: [2, 1],
        3: [3, 2, 1],
        4: [4, 3, 1, 1],
        5: [5, 4, 2, 1, 1],
        6: [6, 5, 3, 2, 1, 1],
        7: [7, 6, 4, 3, 1, 1, 1],
        8: [8, 7, 4, 4, 2, 1, 1, 1],
        9: [9, 8, 5, 4, 2, 2, 1, 1, 1],
        10: [10, 9, 6, 5, 3, 2, 1, 1, 1, 1],
    }

    for n, lst in wikipedia_lexicodes_parameters.items():
        for d, k in enumerate(lst, start=1):
            code = Lexicode(n, d)
            assert code.length == n
            assert code.dimension == k
            assert code.redundancy == n - k
            assert code.minimum_distance() == d


def test_lexicode_error():
    with pytest.raises(ValueError):
        Lexicode(3, 4)
