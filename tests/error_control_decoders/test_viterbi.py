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


def test_viterbi_parallel_transitions(rng):
    # Second input bit has no memory
    convolutional_code = komm.ConvolutionalCode([[0b11, 0b10, 0b0], [0b0, 0b0, 0b1]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, 6)
    decoder = komm.ViterbiDecoder(code)
    u = rng.integers(0, 2, (20, code.dimension))
    np.testing.assert_equal(decoder.decode(code.encode(u)), u)


@pytest.mark.parametrize(
    "convolutional_code, num_blocks",
    [
        (komm.ConvolutionalCode([[0o7, 0o5]]), 6),
        (komm.ConvolutionalCode([[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]]), 3),
        (komm.ConvolutionalCode([[0b11, 0b10, 0b0], [0b0, 0b0, 0b1]]), 3),
        (komm.ConvolutionalCode([[0b11, 0b1]], [0b11]), 6),
    ],
)
@pytest.mark.parametrize("mode", ["direct-truncation", "zero-termination"])
def test_viterbi_exhaustive(convolutional_code, num_blocks, mode, rng):
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks, mode)
    # Soft: no ties, so compare messages
    viterbi = komm.ViterbiDecoder(code, input_type="soft")
    exhaustive = komm.ExhaustiveSearchDecoder(code, input_type="soft")
    r = rng.standard_normal((100, code.length))
    np.testing.assert_equal(viterbi.decode(r), exhaustive.decode(r))
    # Hard: ties are common, so compare distances
    viterbi = komm.ViterbiDecoder(code, input_type="hard")
    exhaustive = komm.ExhaustiveSearchDecoder(code, input_type="hard")
    r = rng.integers(0, 2, (100, code.length))
    v_hat = code.encode(viterbi.decode(r))
    v_ml = exhaustive.decode_to_codeword(r)
    np.testing.assert_equal(
        np.count_nonzero(v_hat != r, axis=-1),
        np.count_nonzero(v_ml != r, axis=-1),
    )
