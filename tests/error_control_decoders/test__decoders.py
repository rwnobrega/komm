import numpy as np
import pytest

import komm
import komm.abc

block = komm.HammingCode(3)
terminated = komm.TerminatedConvolutionalCode(komm.ConvolutionalCode([[0o7, 0o5]]), 12)
bch = komm.BCHCode(4, 5)
reed_muller = komm.ReedMullerCode(1, 5)
spc = komm.SingleParityCheckCode(3)


@pytest.mark.parametrize(
    "code, decoder_class",
    [
        [block, komm.SyndromeTableDecoder],
        [block, komm.ExhaustiveSearchDecoder],
        [terminated, komm.ViterbiDecoder],
        [terminated, komm.BCJRDecoder],
        [bch, komm.BerlekampDecoder],
        [reed_muller, komm.ReedDecoder],
        [spc, komm.WagnerDecoder],
    ],
)
def test_decoders_shapes(code: komm.abc.BlockCode, decoder_class):
    decoder: komm.abc.BlockDecoder = decoder_class(code)
    k, n = code.dimension, code.length
    for b in range(1, 5):
        u_hat = decoder.decode(np.zeros((3, 4, b * n), dtype=int))
        assert u_hat.shape == (3, 4, b * k)
    with pytest.raises(ValueError):
        decoder.decode(np.zeros((3, 4, n + 1), dtype=int))
