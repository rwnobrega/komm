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
reed_muller = komm.ReedMullerCode(5, 1)
spc = komm.SingleParityCheckCode(3)
polar = komm.PolarCode(3, [0, 1, 2, 4])

hard_decoders = [
    komm.SyndromeTableDecoder(block),
    komm.BerlekampDecoder(bch),
    komm.BerlekampDecoder(reed_solomon),
    komm.ExhaustiveCodewordDecoder(block, input_type="hard"),
    komm.ReedDecoder(reed_muller, input_type="hard"),
    komm.ViterbiDecoder(terminated, input_type="hard"),
]

soft_decoders = [
    komm.ExhaustiveCodewordDecoder(block, input_type="soft"),
    komm.ReedDecoder(reed_muller, input_type="soft"),
    komm.ViterbiDecoder(terminated, input_type="soft"),
    komm.ExhaustiveBitwiseDecoder(block),
    komm.BCJRDecoder(terminated),
    komm.WagnerDecoder(spc),
    komm.SCDecoder(polar),
]


def decoding_methods(decoders):
    return [
        (decoder, method)
        for decoder in decoders
        for method in ["decode", "decode_to_codeword"]
        if hasattr(decoder, method)
    ]


@pytest.mark.parametrize(
    "code, decoder_class, dtype",
    [
        (block, komm.SyndromeTableDecoder, int),
        (block, komm.ExhaustiveCodewordDecoder, int),
        (block, komm.ExhaustiveBitwiseDecoder, float),
        (block, komm.GaussianEliminationDecoder, int),
        (block, komm.PeelingDecoder, int),
        (terminated, komm.ViterbiDecoder, int),
        (terminated, komm.BCJRDecoder, float),
        (punctured, komm.ExhaustiveCodewordDecoder, int),
        (punctured, komm.ExhaustiveBitwiseDecoder, float),
        (punctured, komm.ViterbiDecoder, int),
        (punctured, komm.BCJRDecoder, float),
        (punctured, komm.SyndromeTableDecoder, int),
        (bch, komm.BerlekampDecoder, int),
        (reed_solomon, komm.BerlekampDecoder, int),
        (reed_muller, komm.ReedDecoder, int),
        (spc, komm.WagnerDecoder, float),
        (polar, komm.SCDecoder, float),
    ],
)
def test_decoders_shapes(code: komm.abc.BlockCode, decoder_class, dtype):
    decoder: komm.abc.BlockDecoder = decoder_class(code)
    k, n = code.dimension, code.length
    for b in range(1, 5):
        u_hat = decoder.decode(np.zeros((3, 4, b * n), dtype=dtype))
        assert u_hat.shape == (3, 4, b * k)
    with pytest.raises(ValueError, match="last dimension of array must be a multiple"):
        decoder.decode(np.zeros((3, 4, n + 1), dtype=dtype))


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


@pytest.mark.parametrize("decoder, method", decoding_methods(hard_decoders))
@pytest.mark.parametrize("bit", [2, -1])
def test_hard_decoders_invalid_input(decoder, method, bit):
    r = np.zeros(decoder.code.length, dtype=int)
    r[0] = bit
    with pytest.raises(ValueError, match="elements of 'input' must be 0 or 1"):
        getattr(decoder, method)(r)


@pytest.mark.parametrize("decoder, method", decoding_methods(hard_decoders))
def test_hard_decoders_input_not_integer(decoder, method):
    with pytest.raises(TypeError, match="'input' must contain only integers"):
        getattr(decoder, method)(np.zeros(decoder.code.length))


@pytest.mark.parametrize("decoder", hard_decoders)
@pytest.mark.parametrize("dtype", [bool, np.uint8])
def test_hard_decoders_input_dtype(decoder, dtype, rng):
    r = rng.integers(0, 2, decoder.code.length)
    np.testing.assert_array_equal(decoder.decode(r.astype(dtype)), decoder.decode(r))


@pytest.mark.parametrize("decoder, method", decoding_methods(soft_decoders))
@pytest.mark.parametrize("dtype", [int, bool, complex])
def test_soft_decoders_input_not_float(decoder, method, dtype):
    with pytest.raises(TypeError, match="'input' must contain only floats"):
        getattr(decoder, method)(np.zeros(decoder.code.length, dtype=dtype))
