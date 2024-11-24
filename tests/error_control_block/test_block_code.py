import numpy as np

import komm


def test_block_code():
    generator_matrix = [[1, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1]]
    parity_check_matrix = [
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 1],
    ]
    length = 6
    dimension = 2
    redundancy = 4
    rate = 1 / 3
    minimum_distance = 4
    codewords = [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 0],
    ]
    codeword_weight_distribution = [1, 0, 0, 0, 3, 0, 0]
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
    coset_leader_weight_distribution = [1, 6, 7, 2, 0, 0, 0]
    packing_radius = 1
    covering_radius = 3

    code = komm.BlockCode(generator_matrix=generator_matrix)

    assert code.length == length
    assert code.dimension == dimension
    assert code.redundancy == redundancy
    assert code.rate == rate
    assert code.minimum_distance() == minimum_distance
    assert np.array_equal(code.generator_matrix, generator_matrix)
    assert np.array_equal(code.check_matrix, parity_check_matrix)
    assert np.array_equal(code.codewords(), codewords)
    assert np.array_equal(
        code.codeword_weight_distribution(), codeword_weight_distribution
    )
    assert np.array_equal(code.coset_leaders(), coset_leaders)
    assert np.array_equal(
        code.coset_leader_weight_distribution(), coset_leader_weight_distribution
    )
    assert code.covering_radius() == covering_radius
    assert code.packing_radius() == packing_radius
