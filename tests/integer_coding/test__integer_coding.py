import numpy as np
import pytest

import komm
import komm.abc


@pytest.fixture(
    params=[komm.UnaryCode(), komm.FibonacciCode()],
    ids=["UnaryCode", "FibonacciCode"],
)
def code(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize("n", range(1, 100))
def test_integer_coding_constants(code: komm.abc.IntegerCode, n: int):
    for r in range(10):
        message = [n] * r
        assert np.array_equal(code.decode(code.encode(message)), message)


def test_integer_coding_random(code: komm.abc.IntegerCode):
    for _ in range(10):
        message = np.random.randint(1, 100, 100)
        assert np.array_equal(message, code.decode(code.encode(message)))
