import numpy as np
import pytest

import komm
from komm._util.information_theory import random_pmf


def test_fano_code_wikipedia():
    pmf = np.array([15, 7, 6, 6, 5]) / 39
    code = komm.FanoCode(pmf)
    assert code.enc_mapping == {
        (0,): (0, 0),
        (1,): (0, 1),
        (2,): (1, 0),
        (3,): (1, 1, 0),
        (4,): (1, 1, 1),
    }
    np.testing.assert_almost_equal(code.rate(pmf), 89 / 39)


@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("source_block_size", range(1, 4))
def test_fano_code_random_pmf(source_cardinality, source_block_size):
    for _ in range(10):
        pmf = random_pmf(source_cardinality)
        code = komm.FanoCode(pmf, source_block_size)
        assert code.is_uniquely_decodable()
        assert code.is_prefix_free()
        assert code.kraft_parameter() <= 1
        entropy = komm.entropy(pmf)
        min_p = np.min(pmf)
        # For the upper bound below, see [Krajči et al., 2015, apud Wikipedia].
        assert entropy <= code.rate(pmf) <= entropy + (1 - min_p) / source_block_size
        # Permute pmf and check if the rate is the same.
        pmf1 = pmf[np.random.permutation(source_cardinality)]
        code1 = komm.FanoCode(pmf1, source_block_size)
        np.testing.assert_almost_equal(code.rate(pmf), code1.rate(pmf1))
