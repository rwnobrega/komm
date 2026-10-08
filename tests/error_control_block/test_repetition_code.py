from math import comb

import numpy as np
import pytest

import komm


@pytest.mark.parametrize("length", range(2, 11))
def test_repetition_code(length):
    code1 = komm.RepetitionCode(length)
    code2 = komm.BlockCode(generator_matrix=np.ones((1, length), dtype=int))
    np.testing.assert_equal(
        code1.codeword_weight_distribution(),
        code2.codeword_weight_distribution(),
    )
    np.testing.assert_equal(
        code1.coset_leader_weight_distribution(),
        code2.coset_leader_weight_distribution(),
    )


def test_repetition_code_large():
    # Counts exceed 64-bit integers.
    code = komm.RepetitionCode(100)
    distribution = code.coset_leader_weight_distribution()
    assert distribution[49] == comb(100, 49)
    assert sum(distribution) == 2**99


def test_encoder():
    code = komm.RepetitionCode(5)
    np.testing.assert_equal(
        code.encode([[1], [0]]),
        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
    )


def test_repetition_code_invalid_init():
    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        komm.RepetitionCode(0)
    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        komm.RepetitionCode(-1)
    with pytest.raises(TypeError, match="'n' must be an integer"):
        komm.RepetitionCode(3.0)  # type: ignore
    with pytest.raises(TypeError, match="'n' must be an integer"):
        komm.RepetitionCode("3")  # type: ignore
