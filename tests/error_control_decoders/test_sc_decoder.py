import numpy as np

import komm


def test_sc_decoder_kit():
    # "Channel Coding: Graph-based Codes"
    # Video "Polar Codes: Decoding"
    # Karlsruhe Institute of Technology (KIT)
    # Lecturer: Prof. Laurent Schmalen
    # https://youtu.be/m3kWTDHNr7M?t=1610
    code = komm.PolarCode(3, [0, 1, 2, 4])
    decoder = komm.SCDecoder(code)
    np.testing.assert_allclose(
        decoder([1, -4, -3, 2, -2, 3, 4, -1]),
        np.array([-6.85, -5.96, -9.31, -20]),
        atol=0.005,
    )
