import numpy as np
import pytest

import komm


@pytest.mark.parametrize("length,", range(2, 11))
def test_repetition_code(length):
    code1 = komm.RepetitionCode(length)
    code2 = komm.BlockCode(generator_matrix=np.ones((1, length), dtype=int))
    np.testing.assert_array_equal(
        code1.codeword_weight_distribution(),
        code2.codeword_weight_distribution(),
    )
    np.testing.assert_array_equal(
        code1.coset_leader_weight_distribution(),
        code2.coset_leader_weight_distribution(),
    )


def test_encoder():
    code = komm.RepetitionCode(5)
    np.testing.assert_array_equal(
        code.encode([[1], [0]]),
        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
    )
