import numpy as np
import pytest

import komm
from komm._util.matrices import rank


def test_block_code():
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

    code1 = komm.BlockCode(generator_matrix=generator_matrix)
    code2 = komm.BlockCode(check_matrix=check_matrix)

    for code in [code1, code2]:
        np.testing.assert_equal(code.length, 6)
        np.testing.assert_equal(code.dimension, 2)
        np.testing.assert_equal(code.redundancy, 4)
        np.testing.assert_equal(code.rate, 1 / 3)
        np.testing.assert_equal(code.minimum_distance(), 4)
        np.testing.assert_equal(code.generator_matrix, generator_matrix)
        np.testing.assert_equal(code.check_matrix, check_matrix)
        np.testing.assert_equal(code.codewords(), codewords)
        np.testing.assert_equal(
            code.codeword_weight_distribution(), [1, 0, 0, 0, 3, 0, 0]
        )
        np.testing.assert_equal(code.coset_leaders(), coset_leaders)
        np.testing.assert_equal(
            code.coset_leader_weight_distribution(), [1, 6, 7, 2, 0, 0, 0]
        )
        np.testing.assert_equal(code.packing_radius(), 1)
        np.testing.assert_equal(code.covering_radius(), 3)


@pytest.mark.repeat(20)
def test_block_code_mappings():
    while True:
        code = komm.BlockCode(generator_matrix=np.random.randint(0, 2, (4, 8)))
        if rank(code.generator_matrix) == code.dimension:
            break
    k, m = code.dimension, code.redundancy
    for _ in range(100):
        u = np.random.randint(0, 2, (3, 4, k))
        v = code.encode(u)
        np.testing.assert_equal(
            code.inverse_encode(v),
            u,
        )
        np.testing.assert_equal(
            code.check(v),
            np.zeros((3, 4, m)),
        )
