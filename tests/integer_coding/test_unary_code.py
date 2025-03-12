import numpy as np
import pytest

import komm


def test_unary_code():
    message = [0, 1, 2, 3, 4]
    encoded = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    code = komm.UnaryCode()
    np.testing.assert_array_equal(code.encode(message), encoded)
    np.testing.assert_array_equal(code.decode(encoded), message)


@pytest.mark.parametrize("n", range(200))
def test_unary_code_length(n):
    code = komm.UnaryCode()
    encoded = code.encode([n])
    assert len(encoded) == n + 1
