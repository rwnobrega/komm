import numpy as np
import pytest

import komm
import komm.abc

block = komm.HammingCode(3)
terminated = komm.TerminatedConvolutionalCode(
    komm.ConvolutionalCode([[0b111, 0b101]]),
    num_blocks=5,
    mode="zero-termination",
)
bch = komm.BCHCode(4, 5)
reed_muller = komm.ReedMullerCode(1, 5)


@pytest.mark.parametrize(
    "code, decoder_class",
    [
        [block, komm.SyndromeTableDecoder],
        [block, komm.ExhaustiveSearchDecoder],
        [terminated, komm.ViterbiDecoder],
        [terminated, komm.BCJRDecoder],
        [bch, komm.BerlekampDecoder],
        [reed_muller, komm.ReedDecoder],
    ],
)
def test_decoders_shapes(code: komm.abc.BlockCode, decoder_class):
    decoder: komm.abc.BlockDecoder = decoder_class(code)
    k, n = code.dimension, code.length
    u_hat = decoder(np.zeros((3, 4, n), dtype=int))
    assert u_hat.shape == (3, 4, k)
    with np.testing.assert_raises(ValueError):
        decoder(np.zeros((3, 4, n + 1), dtype=int))
