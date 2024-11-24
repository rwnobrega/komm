import numpy as np

import komm


def test_block_code():
    generator_matrix = [
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 1],
    ]
    parity_check_matrix = [
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

    code = komm.BlockCode(generator_matrix=generator_matrix)

    np.testing.assert_equal(code.length, 6)
    np.testing.assert_equal(code.dimension, 2)
    np.testing.assert_equal(code.redundancy, 4)
    np.testing.assert_equal(code.rate, 1 / 3)
    np.testing.assert_equal(code.minimum_distance(), 4)
    np.testing.assert_array_equal(code.generator_matrix, generator_matrix)
    np.testing.assert_array_equal(code.check_matrix, parity_check_matrix)
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
