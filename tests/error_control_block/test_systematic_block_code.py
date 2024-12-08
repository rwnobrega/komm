import numpy as np
import pytest

import komm


def test_systematic_block_code():
    parity_submatrix = [
        [1, 1, 0, 1],
        [0, 1, 1, 1],
    ]
    generator_matrix = [
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 1],
    ]
    check_matrix = [
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 1],
    ]
    codewords = [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 0],
    ]
    coset_leaders = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
    ]

    code = komm.SystematicBlockCode(parity_submatrix=parity_submatrix)

    np.testing.assert_equal(code.length, 6)
    np.testing.assert_equal(code.dimension, 2)
    np.testing.assert_equal(code.redundancy, 4)
    np.testing.assert_equal(code.rate, 1 / 3)
    np.testing.assert_equal(code.minimum_distance(), 4)
    np.testing.assert_array_equal(code.generator_matrix, generator_matrix)
    np.testing.assert_array_equal(code.check_matrix, check_matrix)
    np.testing.assert_array_equal(code.codewords(), codewords)
    np.testing.assert_array_equal(
        code.codeword_weight_distribution(), [1, 0, 0, 0, 3, 0, 0]
    )
    np.testing.assert_array_equal(code.coset_leaders(), coset_leaders)
    np.testing.assert_array_equal(
        code.coset_leader_weight_distribution(), [1, 6, 7, 2, 0, 0, 0]
    )
    np.testing.assert_equal(code.packing_radius(), 1)
    np.testing.assert_equal(code.covering_radius(), 3)


@pytest.mark.parametrize("execution_number", range(20))
def test_systematic_block_code_mappings(execution_number):
    code = komm.SystematicBlockCode(parity_submatrix=np.random.randint(0, 2, (4, 4)))
    k, m = code.dimension, code.redundancy
    for _ in range(100):
        u = np.random.randint(0, 2, (3, 4, k))
        v = code.enc_mapping(u)
        np.testing.assert_array_equal(
            code.inv_enc_mapping(v),
            u,
        )
        np.testing.assert_array_equal(
            code.chk_mapping(v),
            np.zeros((3, 4, m)),
        )


def test_systematic_block_code_mappings_invalid_input():
    code = komm.SystematicBlockCode(
        parity_submatrix=[
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )

    # For 'enc_mapping', last dimension of 'u' should be the code dimension (3)
    code.enc_mapping(np.zeros((3, 4, 3)))  # Correct
    with np.testing.assert_raises(ValueError):
        code.enc_mapping(np.zeros((3, 4, 4)))  # Incorrect

    # For 'inv_enc_mapping', last dimension of 'v' should be the code length (6)
    code.inv_enc_mapping(np.zeros((3, 4, 6)))  # Correct
    with np.testing.assert_raises(ValueError):
        code.inv_enc_mapping(np.zeros((3, 4, 5)))

    # For 'chk_mapping', last dimension of 'r' should be the code length (6)
    code.chk_mapping(np.zeros((3, 4, 6)))  # Correct
    with np.testing.assert_raises(ValueError):
        code.chk_mapping(np.zeros((3, 4, 5)))


def test_block_code_inv_enc_mapping_invalid_input():
    code = komm.SystematicBlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    r = np.zeros(code.length)
    code.inv_enc_mapping(r)  # Correct
    with np.testing.assert_raises(ValueError):
        r[0] = 1
        code.inv_enc_mapping(r)  # Incorrect
