import numpy as np

import komm


def test_block_decoder():
    code = komm.HammingCode(3)
    decoder = komm.BlockDecoder(code)
    np.testing.assert_array_equal(decoder([0, 0, 0, 0, 0, 0, 0]), [0, 0, 0, 0])
