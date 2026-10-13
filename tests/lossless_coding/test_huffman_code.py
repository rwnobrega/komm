import numpy as np
import pytest

import komm

from .util import extended_pmf, random_pmf


def test_huffman_code_sayood_1():
    # [Say06, p. 47]
    pmf = [0.2, 0.4, 0.2, 0.1, 0.1]
    code = komm.HuffmanCode(pmf, policy="high")
    assert code.enc_mapping == {
        0: (1, 0),
        1: (0, 0),
        2: (1, 1),
        3: (0, 1, 0),
        4: (0, 1, 1),
    }
    assert code.average_length(pmf) == 2.2


def test_huffman_code_sayood_2():
    # [Say06, p. 44]
    pmf = [0.2, 0.4, 0.2, 0.1, 0.1]
    code = komm.HuffmanCode(pmf, policy="low")
    assert code.enc_mapping == {
        0: (0, 1),
        1: (1,),
        2: (0, 0, 0),
        3: (0, 0, 1, 0),
        4: (0, 0, 1, 1),
    }
    assert code.average_length(pmf) == 2.2


def test_huffman_code_haykin_1():
    # [Hay04, p. 579]
    pmf = [0.4, 0.2, 0.2, 0.1, 0.1]
    code = komm.HuffmanCode(pmf)
    assert code.enc_mapping == {
        0: (0, 0),
        1: (1, 0),
        2: (1, 1),
        3: (0, 1, 0),
        4: (0, 1, 1),
    }


def test_huffman_code_haykin_2():
    # [Hay04, p. 620, Problem 9.11]
    pmf = [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]
    code1 = komm.HuffmanCode(pmf, policy="high")
    assert code1.enc_mapping == {
        0: (0, 1, 0),
        1: (0, 1, 1),
        2: (0, 0, 0),
        3: (1, 0, 0),
        4: (1, 0, 1),
        5: (0, 0, 1),
        6: (1, 1, 0),
        7: (1, 1, 1),
    }
    assert np.isclose(code1.average_length(pmf), 3.0)
    assert np.isclose(np.var([len(c) for c in code1.enc_mapping.values()]), 0.0)
    code2 = komm.HuffmanCode(pmf, policy="low")
    assert code2.enc_mapping == {
        0: (1, 1, 0),
        1: (1, 1, 1),
        2: (0, 1),
        3: (1, 0, 0),
        4: (1, 0, 1),
        5: (0, 0, 0),
        6: (0, 0, 1, 0),
        7: (0, 0, 1, 1),
    }
    assert np.isclose(code2.average_length(pmf), 3.0)
    assert np.isclose(np.var([len(c) for c in code2.enc_mapping.values()]), 23 / 64)


def test_huffman_code_haykin_3():
    # [Hay04, p. 620, Problem 9.13]
    pmf1 = [0.7, 0.15, 0.15]
    code1 = komm.HuffmanCode(pmf1)
    assert np.isclose(code1.average_length(pmf1), 1.3)
    pmf2 = extended_pmf(pmf1, 2)
    code2 = komm.HuffmanCode(pmf2)
    assert np.isclose(code2.average_length(pmf2) / 2, 1.1975)


def test_huffman_code_haykin_1_canonical():
    # [Hay04, p. 579]
    pmf = [0.4, 0.2, 0.2, 0.1, 0.1]
    code = komm.HuffmanCode(pmf, assignment="canonical")
    assert code.enc_mapping == {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1, 0),
        4: (1, 1, 1),
    }


def test_huffman_code_haykin_2_canonical():
    # [Hay04, p. 620, Problem 9.11]
    pmf = [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]
    code1 = komm.HuffmanCode(pmf, policy="high", assignment="canonical")
    assert code1.enc_mapping == {
        0: (0, 0, 0),
        1: (0, 0, 1),
        2: (0, 1, 0),
        3: (0, 1, 1),
        4: (1, 0, 0),
        5: (1, 0, 1),
        6: (1, 1, 0),
        7: (1, 1, 1),
    }
    assert np.isclose(code1.average_length(pmf), 3.0)
    assert np.isclose(np.var([len(c) for c in code1.enc_mapping.values()]), 0.0)
    code2 = komm.HuffmanCode(pmf, policy="low", assignment="canonical")
    assert code2.enc_mapping == {
        2: (0, 0),
        0: (0, 1, 0),
        1: (0, 1, 1),
        3: (1, 0, 0),
        4: (1, 0, 1),
        5: (1, 1, 0),
        6: (1, 1, 1, 0),
        7: (1, 1, 1, 1),
    }
    assert np.isclose(code2.average_length(pmf), 3.0)
    assert np.isclose(np.var([len(c) for c in code2.enc_mapping.values()]), 23 / 64)


def test_huffman_code_tree_assignment_needs_the_tree():
    # Same lengths, different trees: the 'tree' assignment is not a function of
    # the lengths alone (whereas the 'canonical' one is).
    code1 = komm.HuffmanCode([0.4, 0.2, 0.2, 0.1, 0.1])
    code2 = komm.HuffmanCode([0.2, 0.2, 0.2, 0.2, 0.2])
    assert code1.lengths == code2.lengths
    assert code1.codewords != code2.codewords


def test_huffman_code_invalid_call():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        komm.HuffmanCode([0.5, 0.5, 0.1])
    with pytest.raises(ValueError, match="must be a 1D-array"):
        komm.HuffmanCode([[0.5], [0.5]])
    with pytest.raises(ValueError, match="'pmf' must have size at least 2"):
        komm.HuffmanCode([1.0])
    with pytest.raises(ValueError, match="'policy' must be 'high' or 'low'"):
        komm.HuffmanCode([0.5, 0.5], policy="unknown")  # type: ignore
    with pytest.raises(ValueError, match="'assignment' must be 'tree' or 'canonical'"):
        komm.HuffmanCode([0.5, 0.5], assignment="unknown")  # type: ignore


@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("order", range(1, 4))
def test_huffman_code_policy(source_cardinality, order, rng):
    pmf = extended_pmf(random_pmf(rng, source_cardinality), order)
    code_high = komm.HuffmanCode(pmf, policy="high")
    code_low = komm.HuffmanCode(pmf, policy="low")
    assert np.isclose(code_high.average_length(pmf), code_low.average_length(pmf))


@pytest.mark.parametrize("policy", ["high", "low"])
@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("order", range(1, 4))
def test_huffman_code_complete(policy, source_cardinality, order, rng):
    # Full binary tree, so Kraft equality holds.
    pmf = extended_pmf(random_pmf(rng, source_cardinality), order)
    code = komm.HuffmanCode(pmf, policy=policy)
    assert np.isclose(code.kraft_parameter(), 1.0)


@pytest.mark.parametrize("policy", ["high", "low"])
@pytest.mark.parametrize("source_cardinality", range(2, 7))
@pytest.mark.parametrize("order", range(1, 4))
def test_huffman_code_assignment(policy, source_cardinality, order, rng):
    # Assignment preserves the lengths, and 'canonical' agrees with 'from_lengths'.
    pmf = extended_pmf(random_pmf(rng, source_cardinality), order)
    code_tree = komm.HuffmanCode(pmf, policy=policy)
    code_canonical = komm.HuffmanCode(pmf, policy=policy, assignment="canonical")
    assert code_canonical.lengths == code_tree.lengths
    assert np.isclose(code_canonical.average_length(pmf), code_tree.average_length(pmf))
    expected = komm.SymbolCode.from_lengths(code_tree.lengths)
    assert code_canonical.enc_mapping == expected.enc_mapping
