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
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks, mode)
    decoder = komm.ViterbiDecoder(code, input_type="hard")
    assert np.array_equal(decoder.decode(r), u_hat)


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, mode, r, u_hat",
    [
        (  # Ryan.Lin.09, p. 176--177.
            komm.ConvolutionalCode([[0b111, 0b101]]),
            4,
            "direct-truncation",
            [-0.7, -0.5, -0.8, -0.6, -1.1, +0.4, +0.9, +0.8],
            [1, 0, 0, 0],
        ),
        (  # Abrantes.10, p. 313.
            komm.ConvolutionalCode([[0b111, 0b101]]),
            5,
            "direct-truncation",
            [+0.6, -0.8, -0.3, +0.6, -0.1, -0.1, -0.7, -0.1, -0.6, -0.4],
            [1, 0, 1, 0, 0],
        ),
    ],
)
def test_viterbi_soft(convolutional_code, num_blocks, mode, r, u_hat):
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks, mode)
    decoder = komm.ViterbiDecoder(code, input_type="soft")
    assert np.array_equal(decoder.decode(r), u_hat)
