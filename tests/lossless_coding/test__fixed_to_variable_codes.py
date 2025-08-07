from functools import partial

import numpy as np
import pytest

import komm

from .util import deterministic_pmf, random_pmf, shuffle_pmf


@pytest.fixture(
    params=[
        partial(komm.HuffmanCode, policy="high"),
        partial(komm.HuffmanCode, policy="low"),
        komm.ShannonCode,
        komm.FanoCode,
    ],
    ids=[
        "HuffmanCode-high",
        "HuffmanCode-low",
        "ShannonCode",
        "FanoCode",
    ],
)
def constructor(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize("S", range(2, 7))
@pytest.mark.parametrize("k", range(1, 4))
def test_fixed_to_variable_codes_random_pmf(constructor, S, k):
    for _ in range(10):
        pmf = random_pmf(S)
        entropy = komm.entropy(pmf)
        code: komm.FixedToVariableCode = constructor(pmf, k)
        rate = code.rate(pmf)
        assert code.size == S**k
        assert code.is_uniquely_decodable()
        assert code.is_prefix_free()
        assert code.kraft_parameter() <= 1
        assert entropy <= rate <= entropy + 1 / k
        pmf1 = shuffle_pmf(pmf)
        code1: komm.FixedToVariableCode = constructor(pmf1, k)
        np.testing.assert_allclose(rate, code1.rate(pmf1))


@pytest.mark.parametrize("S", range(2, 7))
@pytest.mark.parametrize("k", range(1, 4))
def test_fixed_to_variable_codes_encode_decode(constructor, S, k):
    pmf = random_pmf(S)
    dms = komm.DiscreteMemorylessSource(pmf)
    code: komm.FixedToVariableCode = constructor(pmf, k)
    x = dms.emit(1000 * k)
    np.testing.assert_equal(x, code.decode(code.encode(x)))


@pytest.mark.parametrize("S", range(2, 7))
@pytest.mark.parametrize("k", range(1, 4))
def test_fixed_to_variable_codes_deterministic(constructor, S, k):
    for i in range(S):
        pmf = deterministic_pmf(S, i)
        code: komm.FixedToVariableCode = constructor(pmf, k)
        np.testing.assert_allclose(code.rate(pmf), 1 / k)
        message = np.full(10 * k, i, dtype=int)
        np.testing.assert_equal(code.decode(code.encode(message)), message)
