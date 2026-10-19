from math import comb

import numpy as np
import pytest

import komm


@pytest.mark.parametrize("length", range(2, 11))
def test_single_parity_check_code(length):
    code1 = komm.SingleParityCheckCode(length)
    code2 = komm.BlockCode(check_matrix=np.ones((1, length), dtype=int))
    np.testing.assert_equal(
        code1.codeword_weight_distribution(),
        code2.codeword_weight_distribution(),
    )
    np.testing.assert_equal(
        code1.coset_leader_weight_distribution(),
        code2.coset_leader_weight_distribution(),
    )


def test_single_parity_check_code_large():
    # Counts exceed 64-bit integers.
    code = komm.SingleParityCheckCode(100)
    distribution = code.codeword_weight_distribution()
    assert distribution[50] == comb(100, 50)
    assert sum(distribution) == 2**99


def test_encoder():
    code = komm.SingleParityCheckCode(5)
    np.testing.assert_equal(
        code.encode([[1, 0, 1, 1], [1, 1, 0, 0]]),
        [[1, 0, 1, 1, 1], [1, 1, 0, 0, 0]],
    )


def test_single_parity_check_code_invalid_init():
    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        komm.SingleParityCheckCode(0)
    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        komm.SingleParityCheckCode(-1)
    with pytest.raises(TypeError, match="'n' must be an integer"):
        komm.SingleParityCheckCode(3.0)  # type: ignore
    with pytest.raises(TypeError, match="'n' must be an integer"):
        komm.SingleParityCheckCode("3")  # type: ignore


@pytest.mark.parametrize("information_set", ["middle", [0, 1, 2, 3]])
def test_single_parity_check_code_invalid_information_set(information_set):
    with pytest.raises(ValueError, match="'information_set' must be 'left' or 'right'"):
        komm.SingleParityCheckCode(5, information_set=information_set)


@pytest.mark.parametrize("n", range(1, 9))
@pytest.mark.parametrize("side, opposite", [("left", "right"), ("right", "left")])
def test_single_parity_check_repetition_duality(n, side, opposite):
    spc = komm.SingleParityCheckCode(n, information_set=side)
    repetition = komm.RepetitionCode(n, information_set=opposite)
    np.testing.assert_equal(spc.generator_matrix, repetition.check_matrix)
    np.testing.assert_equal(spc.check_matrix, repetition.generator_matrix)
