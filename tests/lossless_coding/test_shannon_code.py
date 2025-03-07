import numpy as np
import pytest

import komm
from komm._util.information_theory import random_pmf


def test_shannon_code_wikipedia_1():
    pmf = np.array([15, 7, 6, 6, 5]) / 39
    code = komm.ShannonCode(pmf)
    assert code.enc_mapping == {
        (0,): (0, 0),
        (1,): (0, 1, 0),
        (2,): (0, 1, 1),
        (3,): (1, 0, 0),
        (4,): (1, 0, 1),
    }
    np.testing.assert_almost_equal(code.rate(pmf), 102 / 39)


def test_shannon_code_wikipedia_2():
    pmf = [0.36, 0.18, 0.18, 0.12, 0.09, 0.07]
    code = komm.ShannonCode(pmf)
    assert code.enc_mapping == {
        (0,): (0, 0),
        (1,): (0, 1, 0),
        (2,): (0, 1, 1),
        (3,): (1, 0, 0, 0),
        (4,): (1, 0, 0, 1),
        (5,): (1, 0, 1, 0),
    }


@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("source_block_size", range(1, 4))
def test_shannon_code_random_pmf(source_cardinality, source_block_size):
    for _ in range(10):
        pmf = random_pmf(source_cardinality)
        code = komm.ShannonCode(pmf, source_block_size)
        assert code.size == source_cardinality**source_block_size
        assert code.is_uniquely_decodable()
        assert code.is_prefix_free()
        assert code.kraft_parameter() <= 1
        entropy = komm.entropy(pmf)
        assert entropy <= code.rate(pmf) <= entropy + 1 / source_block_size
        # Permute pmf and check if the rate is the same.
        pmf1 = pmf[np.random.permutation(source_cardinality)]
        code1 = komm.ShannonCode(pmf1, source_block_size)
        np.testing.assert_almost_equal(code.rate(pmf), code1.rate(pmf1))


@pytest.mark.parametrize("source_block_size", [1, 2])
def test_shannon_code_deterministic(source_block_size):
    source_cardinality = 5
    for i in range(source_cardinality):
        pmf = np.zeros(source_cardinality)
        pmf[i] = 1.0
        code = komm.ShannonCode(pmf, source_block_size)
        np.testing.assert_almost_equal(code.rate(pmf), 1 / source_block_size)
        message = np.full(10, i, dtype=int)
        np.testing.assert_array_equal(code.decode(code.encode(message)), message)
