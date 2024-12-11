import numpy as np

import komm


def test_encoder():
    code = komm.SimplexCode(3)
    encoder = komm.BlockEncoder(code)
    np.testing.assert_array_equal(
        encoder([1, 0, 1, 0, 0, 1]),
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    )
