import numpy as np

import komm


def test_encoder():
    code = komm.SimplexCode(3)
    np.testing.assert_array_equal(
        code.encode([[1, 0, 1], [0, 0, 1]]),
        [[1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1]],
    )
