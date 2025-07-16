import numpy as np
import pytest
from typeguard import TypeCheckError

import komm


def test_reed_muller_code_2_4_parameters():
    code = komm.ReedMullerCode(2, 4)
    assert (code.length, code.dimension, code.redundancy) == (16, 11, 5)
    assert code.minimum_distance() == 4


def test_reed_muller_code_2_4_generator_matrix():
    code = komm.ReedMullerCode(2, 4)
    np.testing.assert_equal(
        code.generator_matrix,
        # fmt: off
        komm.int_to_bits(
            [0x8888, 0xA0A0, 0xAA00, 0xC0C0, 0xCC00, 0xF000, 0xAAAA, 0xCCCC, 0xF0F0, 0xFF00, 0xFFFF],
            width=16,
        )
        # fmt: on
    )


def test_reed_muller_code_2_4_weight_distributions():
    code = komm.ReedMullerCode(2, 4)
    np.testing.assert_equal(
        code.codeword_weight_distribution(),
        [1, 0, 0, 0, 140, 0, 448, 0, 870, 0, 448, 0, 140, 0, 0, 0, 1],
    )
    np.testing.assert_equal(
        code.coset_leader_weight_distribution(),
        [1, 16, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )


def test_reed_muller_code_2_4_GH_orthogonality():
    code = komm.ReedMullerCode(2, 4)
    np.testing.assert_equal(
        np.dot(code.generator_matrix, code.check_matrix.T) % 2,
        np.zeros((code.dimension, code.redundancy), dtype=int),
    )


def test_reed_muller_code_2_4_reed_partitions():
    code = komm.ReedMullerCode(2, 4)
    reed_partitions = code.reed_partitions()
    np.testing.assert_equal(
        reed_partitions[0],
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    )
    np.testing.assert_equal(
        reed_partitions[1],
        [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]],
    )
    np.testing.assert_equal(
        reed_partitions[8],
        [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]],
    )


def test_reed_muller_code_2_4_encoder():
    code = komm.ReedMullerCode(1, 5)
    np.testing.assert_equal(
        code.encode([0, 0, 0, 0, 0, 1]),
        np.ones(32, dtype=int),
    )


def test_reed_muller_code_invalid_init():
    with pytest.raises(ValueError, match="'rho' and 'mu' must satisfy 0 <= rho < mu"):
        komm.ReedMullerCode(2, 2)
    with pytest.raises(ValueError, match="'rho' and 'mu' must satisfy 0 <= rho < mu"):
        komm.ReedMullerCode(-1, 3)
    with pytest.raises(ValueError, match="'rho' and 'mu' must satisfy 0 <= rho < mu"):
        komm.ReedMullerCode(3, 2)
    with pytest.raises(TypeCheckError):
        komm.ReedMullerCode(1.5, 3)  # type: ignore
