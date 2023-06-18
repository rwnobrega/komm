import numpy as np
import pytest

import komm


@pytest.mark.parametrize("length,", range(2, 11))
def test_repetition_code(length):
    code1 = komm.RepetitionCode(length)
    code2 = komm.BlockCode.from_generator_matrix(np.ones((1, length), dtype=int))
    assert np.array_equal(code1.codeword_weight_distribution, code2.codeword_weight_distribution)
    assert np.array_equal(code1.coset_leader_weight_distribution, code2.coset_leader_weight_distribution)