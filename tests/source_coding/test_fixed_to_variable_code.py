import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "codewords, x_output, x_rate",
    [
        (
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)],
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            3.0,
        ),
        (
            [(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            2.2,
        ),
    ],
)
def test_fixed_to_variable_code(codewords, x_output, x_rate):
    code = komm.FixedToVariableCode(codewords)
    x = [3, 0, 1, 1, 1, 0, 2, 0]
    pmf = [0.4, 0.2, 0.2, 0.1, 0.1]
    assert np.array_equal(code.encode(x), x_output)
    assert np.array_equal(code.decode(x_output), x)
    assert np.isclose(code.rate(pmf), x_rate)


@pytest.mark.parametrize(
    "codewords",
    [
        [(0, 0), (0, 0)],
        [(0,), (0, 0), (1, 0)],
        [(0, 0), (0,), (1, 0)],
        [(0, 0), (1, 0), (0,)],
    ],
)
def test_fixed_to_variable_code_not_prefix_free(codewords):
    with pytest.raises(ValueError):
        komm.FixedToVariableCode(codewords)


@pytest.mark.parametrize(
    "pmf",
    [
        [0.5, 0.5, 0.1],
        [-0.4, 0.4, 1.0],
    ],
)
def test_fixed_to_variable_code_invalid_pmf(pmf):
    code = komm.FixedToVariableCode([(0,), (1, 0), (1, 1)])
    with pytest.raises(ValueError):
        code.rate(pmf)
