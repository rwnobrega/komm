import numpy as np

import komm


def test_fixed_to_variable_code():
    code1 = komm.FixedToVariableCode([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0)])
    code2 = komm.FixedToVariableCode([(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)])
    x = [3, 0, 1, 1, 1, 0, 2, 0]
    y1 = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y2 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    pmf = [0.4, 0.2, 0.2, 0.1, 0.1]

    assert np.array_equal(code1.encode(x), y1)
    assert np.array_equal(code1.decode(y1), x)
    assert np.array_equal(code2.encode(x), y2)
    assert np.array_equal(code2.decode(y2), x)
    assert np.isclose(code1.rate(pmf), 3.0)
    assert np.isclose(code2.rate(pmf), 2.2)
