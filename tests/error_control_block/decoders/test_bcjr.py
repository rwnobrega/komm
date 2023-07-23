import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, mode, decoder_kwargs, r, u_hat",
    [
        (  # Abrantes.10, p. 434--437.
            komm.ConvolutionalCode([[0b111, 0b101]]),
            4,
            "zero-termination",
            {"snr": 1.25},
            -np.array([+0.3, +0.1, -0.5, +0.2, +0.8, +0.5, -0.5, +0.3, +0.1, -0.7, +1.5, -0.4]),
            -np.array([1.78, 0.24, -1.97, 5.52]),
        ),
        (  # Lin.Costello.04, p. 572--575.
            komm.ConvolutionalCode([[0b11, 0b1]], [0b11]),
            3,
            "zero-termination",
            {"snr": 0.25},
            -np.array([+0.8, +0.1, +1.0, -0.5, -1.8, +1.1, +1.6, -1.6]),
            -np.array([0.48, 0.62, -1.02]),
        ),
    ],
)
def test_bcjr(convolutional_code, num_blocks, mode, decoder_kwargs, r, u_hat):
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=num_blocks, mode=mode)
    decoder = komm.BlockDecoder(code, method="bcjr", decoder_kwargs=decoder_kwargs)
    assert np.allclose(decoder(r), u_hat, atol=0.05)
