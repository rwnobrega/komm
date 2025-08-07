import numpy as np
import pytest

import komm

from .util import random_pmf


def test_huffman_code_sayood_1():
    # [Say06, p. 47]
    pmf = [0.2, 0.4, 0.2, 0.1, 0.1]
    code = komm.HuffmanCode(pmf, policy="high")
    assert code.enc_mapping == {
        (0,): (1, 0),
        (1,): (0, 0),
        (2,): (1, 1),
        (3,): (0, 1, 0),
        (4,): (0, 1, 1),
    }
    assert code.rate(pmf) == 2.2


def test_huffman_code_sayood_2():
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


def test_huffman_code_haykin_1():
    # [Hay04, p. 579]
    pmf = [0.4, 0.2, 0.2, 0.1, 0.1]
    code = komm.HuffmanCode(pmf)
    assert code.enc_mapping == {
        (0,): (0, 0),
        (1,): (1, 0),
        (2,): (1, 1),
        (3,): (0, 1, 0),
        (4,): (0, 1, 1),
    }


def test_huffman_code_haykin_2():
    # [Hay04, p. 620, Problem 9.11]
    pmf = [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]
    code1 = komm.HuffmanCode(pmf, policy="high")
    assert code1.enc_mapping == {
        (0,): (0, 1, 0),
        (1,): (0, 1, 1),
        (2,): (0, 0, 0),
        (3,): (1, 0, 0),
        (4,): (1, 0, 1),
        (5,): (0, 0, 1),
        (6,): (1, 1, 0),
        (7,): (1, 1, 1),
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


def test_huffman_code_haykin_3():
    # [Hay04, p. 620, Problem 9.13]
    pmf = [0.7, 0.15, 0.15]
    code1 = komm.HuffmanCode(pmf)
    assert np.isclose(code1.rate(pmf), 1.3)
    code2 = komm.HuffmanCode(pmf, source_block_size=2)
    assert np.isclose(code2.rate(pmf), 1.1975)


def test_huffman_code_invalid_call():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        komm.HuffmanCode([0.5, 0.5, 0.1])
    with pytest.raises(ValueError, match="must be a 1D-array"):
        komm.HuffmanCode([[0.5], [0.5]])
    with pytest.raises(ValueError, match="must be in"):
        komm.HuffmanCode([0.5, 0.5], policy="unknown")  # type: ignore


@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("source_block_size", range(1, 4))
def test_huffman_code_policy(source_cardinality, source_block_size):
    pmf = random_pmf(source_cardinality)
    code_high = komm.HuffmanCode(pmf, source_block_size, policy="high")
    code_low = komm.HuffmanCode(pmf, source_block_size, policy="low")
    assert np.isclose(code_high.rate(pmf), code_low.rate(pmf))
