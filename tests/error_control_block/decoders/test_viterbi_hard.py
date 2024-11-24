import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, mode, r, u_hat",
    [
        (  # Lin.Costello.04, p. 522--523.
            komm.ConvolutionalCode([[0b011, 0b101, 0b111]]),
            5,
            "zero-termination",
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1],
        ),
        (  # Abrantes.10, p. 307.
            komm.ConvolutionalCode([[0b111, 0b101]]),
            10,
            "direct-truncation",
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        ),
    ],
)
def test_viterbi_hard(convolutional_code, num_blocks, mode, r, u_hat):
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks=num_blocks, mode=mode
    )
    decoder = komm.BlockDecoder(code, method="viterbi-hard")
    assert np.array_equal(decoder(r), u_hat)
