from math import ceil, log2

import numpy as np
import pytest

import komm
from komm._util.information_theory import random_pmf


def test_tunstall_code():
    # Sayood.06, p. 71.
    pmf = [0.6, 0.3, 0.1]
    code = komm.TunstallCode(pmf, 3)
    assert code.inv_dec_mapping == {
        (0, 0, 0): (0, 0, 0),
        (0, 0, 1): (0, 0, 1),
        (0, 0, 2): (0, 1, 0),
        (0, 1): (0, 1, 1),
        (0, 2): (1, 0, 0),
        (1,): (1, 0, 1),
        (2,): (1, 1, 0),
    }
    assert np.isclose(code.rate(pmf), 3 / 1.96)
    assert code.is_fully_covering()


def test_tunstall_code_invalid_init():
    with pytest.raises(ValueError):
        komm.TunstallCode([0.5, 0.5, 0.1], 3)
    with pytest.raises(ValueError):
        komm.TunstallCode([0.5, 0.5], 0)


@pytest.mark.parametrize("source_cardinality", range(2, 9))
@pytest.mark.parametrize("target_block_size", range(1, 7))
def test_tunstall_code_random_pmf(source_cardinality, target_block_size):
    if 2**target_block_size < source_cardinality:  # target block size too low
        return
    for _ in range(10):
        pmf = random_pmf(source_cardinality)
        code = komm.TunstallCode(pmf, target_block_size)
        assert code.size <= 2**target_block_size
        assert code.is_fully_covering()
        assert code.is_uniquely_encodable()
        assert code.is_prefix_free()
        assert code.rate(pmf) >= komm.entropy(pmf)
        # Permute pmf and check if the rate is the same.
        pmf1 = pmf[np.random.permutation(source_cardinality)]
        code1 = komm.TunstallCode(pmf1, target_block_size)
        np.testing.assert_almost_equal(code.rate(pmf), code1.rate(pmf1))


@pytest.mark.parametrize("source_cardinality", range(2, 7))
def test_tunstall_code_rate_upper_bound(source_cardinality):
    # From MIT 6.441 Supplementary Notes 1, 2/10/94, eq. (5).
    for _ in range(10):
        pmf = random_pmf(source_cardinality)
        min_p = np.min(pmf)
        target_block_size = ceil(log2(1 / min_p)) + 1
        code = komm.TunstallCode(pmf, target_block_size)
        size = code.size
        entropy = komm.entropy(pmf)
        bound = entropy * log2(size + source_cardinality - 2) / log2(size * min_p)
        assert code.rate(pmf) <= bound


@pytest.mark.parametrize("source_cardinality", range(2, 9))
@pytest.mark.parametrize("target_block_size", range(1, 7))
def test_tunstall_code_encode_decode(source_cardinality, target_block_size):
    if 2**target_block_size < source_cardinality:  # target block size too low
        return
    pmf = random_pmf(source_cardinality)
    dms = komm.DiscreteMemorylessSource(pmf)
    code = komm.TunstallCode(pmf, target_block_size=target_block_size)
    x = dms(1000)
    x_hat = code.decode(code.encode(x))[: len(x)]
    assert np.array_equal(x_hat, x)
