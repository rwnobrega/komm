import numpy as np
import pytest

import komm


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


@pytest.mark.parametrize("source_cardinality", range(2, 10))
@pytest.mark.parametrize("target_block_size", range(1, 7))
def test_random_tunstall_code(source_cardinality, target_block_size):
    if 2**target_block_size < source_cardinality:  # target block size too low
        return
    for _ in range(10):
        pmf = np.random.rand(source_cardinality)
        pmf /= pmf.sum()
        code = komm.TunstallCode(pmf, target_block_size)
        assert code.is_prefix_free()
        assert code.is_fully_covering()


@pytest.mark.parametrize("source_cardinality", range(2, 10))
@pytest.mark.parametrize("target_block_size", range(1, 7))
def test_tunstall_code_encode_decode(source_cardinality, target_block_size):
    if 2**target_block_size < source_cardinality:  # target block size too low
        return
    integers = np.random.randint(0, 100, source_cardinality)
    pmf = integers / integers.sum()
    dms = komm.DiscreteMemorylessSource(pmf)
    code = komm.TunstallCode(pmf, target_block_size=target_block_size)
    x = dms(1000)
    x_hat = code.decode(code.encode(x))[: len(x)]
    assert np.array_equal(x_hat, x)
