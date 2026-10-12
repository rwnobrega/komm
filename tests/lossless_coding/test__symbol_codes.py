from functools import partial

import numpy as np
import pytest

import komm
import komm.abc

from .util import deterministic_pmf, extended_pmf, random_pmf


@pytest.fixture(
    params=[
        komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)]),
        komm.HuffmanCode([0.2, 0.4, 0.2, 0.1, 0.1]),
        komm.FanoCode([0.2, 0.4, 0.2, 0.1, 0.1]),
        komm.ShannonCode([0.2, 0.4, 0.2, 0.1, 0.1]),
    ],
    ids=lambda code: type(code).__name__,
)
def code(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(
    params=[
        partial(komm.HuffmanCode, policy="high", assignment="tree"),
        partial(komm.HuffmanCode, policy="low", assignment="tree"),
        partial(komm.HuffmanCode, policy="high", assignment="canonical"),
        partial(komm.HuffmanCode, policy="low", assignment="canonical"),
        partial(komm.FanoCode, assignment="tree"),
        partial(komm.FanoCode, assignment="canonical"),
        komm.ShannonCode,
    ],
    ids=[
        "HuffmanCode-high-tree",
        "HuffmanCode-low-tree",
        "HuffmanCode-high-canonical",
        "HuffmanCode-low-canonical",
        "FanoCode-tree",
        "FanoCode-canonical",
        "ShannonCode",
    ],
)
def constructor(request: pytest.FixtureRequest):
    return request.param


def test_symbol_codes_instance(code: komm.abc.SymbolCode):
    assert isinstance(code, komm.abc.SymbolCode)


def test_symbol_codes_encode_decode(code: komm.abc.SymbolCode, rng):
    x = rng.integers(0, code.source_cardinality, 1000)
    np.testing.assert_equal(code.decode(code.encode(x)), x)


@pytest.mark.parametrize("S", range(2, 7))
@pytest.mark.parametrize("k", range(1, 4))
def test_symbol_codes_random_pmf(constructor, S, k, rng):
    # Extended source of order k
    for _ in range(10):
        pmf = random_pmf(rng, S)
        entropy = komm.entropy(pmf)
        pmf_k = extended_pmf(pmf, k)
        code: komm.abc.SymbolCode = constructor(pmf_k)
        rate = code.average_length(pmf_k) / k
        assert code.is_uniquely_decodable()
        assert code.is_prefix_free()
        assert code.kraft_parameter() <= 1
        assert entropy <= rate <= entropy + 1 / k
        pmf1_k = extended_pmf(rng.permutation(pmf), k)
        code1: komm.abc.SymbolCode = constructor(pmf1_k)
        np.testing.assert_allclose(rate, code1.average_length(pmf1_k) / k)


@pytest.mark.parametrize("S", range(2, 7))
@pytest.mark.parametrize("k", range(1, 4))
def test_symbol_codes_random_encode_decode(constructor, S, k, rng):
    pmf = extended_pmf(random_pmf(rng, S), k)
    dms = komm.DiscreteMemorylessSource(pmf)
    code: komm.abc.SymbolCode = constructor(pmf)
    x = dms.emit(1000)
    np.testing.assert_equal(x, code.decode(code.encode(x)))


@pytest.mark.parametrize("S", range(2, 7))
def test_symbol_codes_deterministic(constructor, S):
    if constructor is komm.ShannonCode:
        return
    for i in range(S):
        pmf = deterministic_pmf(S, i)
        code: komm.abc.SymbolCode = constructor(pmf)
        np.testing.assert_allclose(code.average_length(pmf), 1)
        message = np.full(10, i)
        np.testing.assert_equal(code.decode(code.encode(message)), message)
