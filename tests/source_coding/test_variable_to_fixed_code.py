import numpy as np

import komm


def test_variable_to_fixed_code():
    # Sayood.06, p. 69.
    code = komm.VariableToFixedCode([(0, 0, 0), (0, 0, 1), (0, 1), (1,)])
    assert np.array_equal(
        code.encode([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]), [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
    )
    assert np.array_equal(
        code.decode([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]), [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    )
    assert np.isclose(code.rate([2 / 3, 1 / 3]), 18 / 19)
