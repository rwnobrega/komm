import numpy as np
import pytest

import komm


def test_variable_to_fixed_code():
    # Sayood.06, p. 69.
    code = komm.VariableToFixedCode([(0, 0, 0), (0, 0, 1), (0, 1), (1,)])
    x = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    y = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
    assert np.array_equal(code.encode(x), y)
    assert np.array_equal(code.decode(y), x)
    assert np.isclose(code.rate([2 / 3, 1 / 3]), 18 / 19)


@pytest.mark.parametrize(
    "sourcewords",
    [
        [(0, 0), (0, 0)],
        [(0,), (0, 0), (1, 0)],
        [(0, 0), (0,), (1, 0)],
        [(0, 0), (1, 0), (0,)],
    ],
)
def test_variable_to_fixed_code_not_prefix_free(sourcewords):
    with pytest.raises(ValueError):
        komm.VariableToFixedCode(sourcewords)


@pytest.mark.parametrize(
    "pmf",
    [
        [0.5, 0.5, 0.1],
        [-0.4, 0.4, 1.0],
    ],
)
def test_variable_to_fixed_code_invalid_pmf(pmf):
    code = komm.VariableToFixedCode([(0, 0, 0), (0, 0, 1), (0, 1), (1,)])
    with pytest.raises(ValueError):
        code.rate(pmf)
