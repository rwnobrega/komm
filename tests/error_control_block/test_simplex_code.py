import numpy as np

import komm


def test_encoder():
    code = komm.SimplexCode(3)
    encoder = komm.BlockEncoder(code)
    np.testing.assert_array_equal(
        encoder([1, 0, 1, 0, 0, 1]),
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    )


def test_decoder():
    code = komm.SimplexCode(3)
    decoder = komm.BlockDecoder(code)
    np.testing.assert_array_equal(
        decoder([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]),
        [1, 0, 1, 0, 0, 0],
    )
