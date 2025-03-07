import numpy as np
import pytest

import komm
from komm._util.information_theory import random_pmf


def test_huffman_code_1():
    # [Say06, p. 47]
    pmf = [0.2, 0.4, 0.2, 0.1, 0.1]
    code = komm.HuffmanCode(pmf)
    assert code.enc_mapping == {
        (0,): (1, 1),
        (1,): (0, 0),
        (2,): (1, 0),
        (3,): (0, 1, 1),
        (4,): (0, 1, 0),
    }
    assert code.rate(pmf) == 2.2


def test_huffman_code_2():
    # [Say06, p. 44]
    pmf = [0.2, 0.4, 0.2, 0.1, 0.1]
    code = komm.HuffmanCode(pmf, policy="low")
    assert code.enc_mapping == {
        (0,): (0, 1),
        (1,): (1,),
        (2,): (0, 0, 0),
        (3,): (0, 0, 1, 0),
        (4,): (0, 0, 1, 1),
    }
    assert code.rate(pmf) == 2.2


def test_huffman_code_3():
    # [Hay04, p. 620]
    pmf = [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]
    code1 = komm.HuffmanCode(pmf, policy="high")
    assert code1.enc_mapping == {
        (0,): (1, 1, 1),
        (1,): (1, 1, 0),
        (2,): (0, 0, 1),
        (3,): (1, 0, 1),
        (4,): (1, 0, 0),
        (5,): (0, 0, 0),
        (6,): (0, 1, 1),
        (7,): (0, 1, 0),
    }
    assert np.isclose(code1.rate(pmf), 3.0)
    assert np.isclose(np.var([len(c) for c in code1.enc_mapping.values()]), 0.0)
    code2 = komm.HuffmanCode([0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1], policy="low")
    assert code2.enc_mapping == {
        (0,): (1, 1, 0),
        (1,): (1, 1, 1),
        (2,): (0, 1),
        (3,): (1, 0, 0),
        (4,): (1, 0, 1),
        (5,): (0, 0, 0),
        (6,): (0, 0, 1, 0),
        (7,): (0, 0, 1, 1),
    }
    assert np.isclose(code2.rate(pmf), 3.0)
    assert np.isclose(np.var([len(c) for c in code2.enc_mapping.values()]), 23 / 64)


def test_huffman_code_4():
    # [Hay04, p. 620]
    pmf = [0.7, 0.15, 0.15]
    code1 = komm.HuffmanCode(pmf, source_block_size=1)
    assert np.isclose(code1.rate(pmf), 1.3)
    code2 = komm.HuffmanCode([0.7, 0.15, 0.15], source_block_size=2)
    assert np.isclose(code2.rate(pmf), 1.1975)


def test_huffman_code_invalid_call():
    with pytest.raises(ValueError):
        komm.HuffmanCode([0.5, 0.5, 0.1])
    with pytest.raises(ValueError):
        komm.HuffmanCode([0.5, 0.5], source_block_size=0)
    with pytest.raises(ValueError):
        komm.HuffmanCode([0.5, 0.5], policy="unknown")  # type: ignore


@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("source_block_size", range(1, 4))
@pytest.mark.parametrize("policy", ["high", "low"])
def test_huffman_code_random_pmf(source_cardinality, source_block_size, policy):
    for _ in range(10):
        pmf = random_pmf(source_cardinality)
        code = komm.HuffmanCode(pmf, source_block_size=source_block_size, policy=policy)
        assert code.size == source_cardinality**source_block_size
        assert code.is_uniquely_decodable()
        assert code.is_prefix_free()
        assert code.kraft_parameter() <= 1
        entropy = komm.entropy(pmf)
        assert entropy <= code.rate(pmf) <= entropy + 1 / source_block_size
        # Permute pmf and check if the rate is the same.
        pmf1 = pmf[np.random.permutation(source_cardinality)]
        code1 = komm.HuffmanCode(pmf1, source_block_size, policy)
        np.testing.assert_almost_equal(code.rate(pmf), code1.rate(pmf1))


@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("source_block_size", range(1, 4))
@pytest.mark.parametrize("policy", ["high", "low"])
def test_huffman_code_encode_decode(source_cardinality, source_block_size, policy):
    pmf = random_pmf(source_cardinality)
    dms = komm.DiscreteMemorylessSource(pmf)
    code = komm.HuffmanCode(pmf, source_block_size=source_block_size, policy=policy)
    x = dms(1000 * source_block_size)
    x_hat = code.decode(code.encode(x))
    assert np.array_equal(x_hat, x)


@pytest.mark.parametrize("source_block_size", [1, 2])
@pytest.mark.parametrize("policy", ["high", "low"])
def test_huffman_code_deterministic(source_block_size, policy):
    source_cardinality = 5
    for i in range(source_cardinality):
        pmf = np.zeros(source_cardinality)
        pmf[i] = 1.0
        code = komm.HuffmanCode(pmf, source_block_size, policy)
        np.testing.assert_almost_equal(code.rate(pmf), 1 / source_block_size)
