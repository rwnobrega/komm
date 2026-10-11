import numpy as np
import pytest

import komm
import komm.abc


@pytest.fixture(
    params=[
        komm.SymbolCode.from_codewords([(0,), (1, 0), (1, 1)]),
    ],
    ids=lambda code: type(code).__name__,
)
def code(request: pytest.FixtureRequest):
    return request.param


def test_symbol_codes_instance(code: komm.abc.SymbolCode):
    assert isinstance(code, komm.abc.SymbolCode)


def test_symbol_codes_encode_decode(code: komm.abc.SymbolCode, rng):
    x = rng.integers(0, code.source_cardinality, 1000)
    np.testing.assert_equal(code.decode(code.encode(x)), x)
