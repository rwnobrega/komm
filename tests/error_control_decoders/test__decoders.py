import numpy as np
import pytest

import komm
import komm.abc

block = komm.HammingCode(3)
terminated = komm.TerminatedConvolutionalCode(komm.ConvolutionalCode([[0o7, 0o5]]), 12)
punctured = komm.TerminatedConvolutionalCode(
    komm.ConvolutionalCode([[0o7, 0o5]]),
    num_blocks=12,
    puncturing_matrix=[[1, 1], [1, 0]],
)
bch = komm.BCHCode(4, 5)
reed_solomon = komm.ReedSolomonCode(3, 5)
reed_muller = komm.ReedMullerCode(1, 5)
spc = komm.SingleParityCheckCode(3)
polar = komm.PolarCode(3, [0, 1, 2, 4])

hard_decoders = [
    (block, komm.SyndromeTableDecoder),
    (bch, komm.BerlekampDecoder),
    (reed_solomon, komm.BerlekampDecoder),
]


@pytest.mark.parametrize(
    "code, decoder_class",
    [
        (block, komm.SyndromeTableDecoder),
        (block, komm.ExhaustiveCodewordDecoder),
        (block, komm.ExhaustiveBitwiseDecoder),
        (block, komm.GaussianEliminationDecoder),
        (block, komm.PeelingDecoder),
        (terminated, komm.ViterbiDecoder),
        (terminated, komm.BCJRDecoder),
        (punctured, komm.ExhaustiveCodewordDecoder),
        (punctured, komm.ExhaustiveBitwiseDecoder),
        (punctured, komm.ViterbiDecoder),
        (punctured, komm.BCJRDecoder),
        (punctured, komm.SyndromeTableDecoder),
        (bch, komm.BerlekampDecoder),
        (reed_solomon, komm.BerlekampDecoder),
        (reed_muller, komm.ReedDecoder),
        (spc, komm.WagnerDecoder),
        (polar, komm.SCDecoder),
    ],
)
def test_decoders_shapes(code: komm.abc.BlockCode, decoder_class):
    decoder: komm.abc.BlockDecoder = decoder_class(code)
    k, n = code.dimension, code.length
    for b in range(1, 5):
        u_hat = decoder.decode(np.zeros((3, 4, b * n), dtype=int))
        assert u_hat.shape == (3, 4, b * k)
    with pytest.raises(ValueError, match="last dimension of array must be a multiple"):
        decoder.decode(np.zeros((3, 4, n + 1), dtype=int))


@pytest.mark.parametrize(
    "code, decoder_class, parameter",
    [
        (block, komm.ExhaustiveCodewordDecoder, "input_type"),
        (block, komm.ExhaustiveBitwiseDecoder, "output_type"),
        (terminated, komm.ViterbiDecoder, "input_type"),
        (terminated, komm.BCJRDecoder, "output_type"),
        (reed_muller, komm.ReedDecoder, "input_type"),
        (polar, komm.SCDecoder, "output_type"),
    ],
)
def test_decoders_invalid_type(code, decoder_class, parameter):
    with pytest.raises(ValueError, match=f"'{parameter}' must be 'hard' or 'soft'"):
        decoder_class(code, **{parameter: "sotf"})


@pytest.mark.parametrize("code, decoder_class", hard_decoders)
@pytest.mark.parametrize("method", ["decode", "decode_to_codeword"])
@pytest.mark.parametrize("bit", [2, -1])
def test_hard_decoders_invalid_input(code, decoder_class, method, bit):
    decoder = decoder_class(code)
    r = np.zeros(code.length, dtype=int)
    r[0] = bit
    with pytest.raises(ValueError, match="elements of 'input' must be 0 or 1"):
        getattr(decoder, method)(r)


@pytest.mark.parametrize("code, decoder_class", hard_decoders)
@pytest.mark.parametrize("method", ["decode", "decode_to_codeword"])
def test_hard_decoders_input_not_integer(code, decoder_class, method):
    decoder = decoder_class(code)
    with pytest.raises(TypeError, match="'input' must contain only integers"):
        getattr(decoder, method)(np.zeros(code.length))
