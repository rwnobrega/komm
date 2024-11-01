import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, mode, r, u_hat",
    [
        (  # Ryan.Lin.09, p. 176--177.
            komm.ConvolutionalCode([[0b111, 0b101]]),
            4,
            "direct-truncation",
            np.array([-0.7, -0.5, -0.8, -0.6, -1.1, +0.4, +0.9, +0.8]),
            [1, 0, 0, 0],
        ),
        (  # Abrantes.10, p. 313.
            komm.ConvolutionalCode([[0b111, 0b101]]),
            5,
            "direct-truncation",
            -np.array([-0.6, +0.8, +0.3, -0.6, +0.1, +0.1, +0.7, +0.1, +0.6, +0.4]),
            [1, 0, 1, 0, 0],
        ),
    ],
)
def test_viterbi_soft(convolutional_code, num_blocks, mode, r, u_hat):
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks=num_blocks, mode=mode
    )
    decoder = komm.BlockDecoder(code, method="viterbi_soft")
    assert np.array_equal(decoder(r), u_hat)
