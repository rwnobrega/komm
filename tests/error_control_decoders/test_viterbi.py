import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, mode, r, u_hat",
    [
        (  # [LC04, pp. 522–523]
            komm.ConvolutionalCode([[0b011, 0b101, 0b111]]),
            5,
            "zero-termination",
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1],
        ),
        (  # [Abr10, p. 307]
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
        (  # [RL09, pp. 176–177]
            komm.ConvolutionalCode([[0b111, 0b101]]),
            4,
            "direct-truncation",
            [-0.7, -0.5, -0.8, -0.6, -1.1, +0.4, +0.9, +0.8],
            [1, 0, 0, 0],
        ),
        (  # [Abr10, p. 313]
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


def check_against_exhaustive(code, rng):
    # Soft: no ties, so compare messages
    viterbi = komm.ViterbiDecoder(code, input_type="soft")
    exhaustive = komm.ExhaustiveCodewordDecoder(code, input_type="soft")
    r = rng.standard_normal((100, code.length))
    np.testing.assert_equal(viterbi.decode(r), exhaustive.decode(r))
    np.testing.assert_equal(
        viterbi.decode_to_codeword(r),
        exhaustive.decode_to_codeword(r),
    )
    # Hard: ties are common, so compare distances
    viterbi = komm.ViterbiDecoder(code, input_type="hard")
    exhaustive = komm.ExhaustiveCodewordDecoder(code, input_type="hard")
    r = rng.integers(0, 2, (100, code.length))
    v_hat = viterbi.decode_to_codeword(r)
    v_ml = exhaustive.decode_to_codeword(r)
    np.testing.assert_equal(
        np.count_nonzero(v_hat != r, axis=-1),
        np.count_nonzero(v_ml != r, axis=-1),
    )


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, puncturing_matrix",
    [
        (
            komm.ConvolutionalCode([[0o7, 0o5]]),
            6,
            None,
        ),
        (
            komm.ConvolutionalCode([[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]]),
            3,
            None,
        ),
        (
            komm.ConvolutionalCode([[0b11, 0b10, 0b0], [0b0, 0b0, 0b1]]),
            3,
            None,
        ),
        (
            komm.ConvolutionalCode([[0b11, 0b1]], [0b11]),
            6,
            None,
        ),
        (
            komm.ConvolutionalCode([[0o5, 0o7]]),
            6,
            [[1, 0], [1, 1]],
        ),
        (
            komm.ConvolutionalCode([[0o7, 0o5, 0o0], [0o0, 0o7, 0o5]]),
            4,
            [[1, 1], [1, 1], [1, 0]],
        ),
        (
            komm.ConvolutionalCode([[0o7, 0o5]], [0o7]),
            6,
            [[1, 0], [1, 1]],
        ),
        # Free tail bits
        (
            komm.ConvolutionalCode([[0o7, 0o5, 0o0], [0o0, 0o3, 0o2]]),
            3,
            None,
        ),
        (
            komm.ConvolutionalCode([[0o7, 0o5, 0o0], [0o0, 0o3, 0o2]], [0o5, 0o3]),
            3,
            None,
        ),
        (
            komm.ConvolutionalCode([[0o7, 0o5, 0o0], [0o0, 0o3, 0o2]]),
            4,
            [[1, 1], [1, 1], [1, 0]],
        ),
        (
            komm.HighRateConvolutionalCode([0o6, 0o7, 0o5, 0o1]),
            3,
            None,
        ),
    ],
)
@pytest.mark.parametrize("mode", ["direct-truncation", "zero-termination"])
def test_viterbi_exhaustive(
    convolutional_code, num_blocks, puncturing_matrix, mode, rng
):
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks, mode, puncturing_matrix
    )
    check_against_exhaustive(code, rng)


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, puncturing_matrix",
    [
        (
            komm.ConvolutionalCode([[0o7, 0o5]]),
            6,
            None,
        ),
        (
            komm.ConvolutionalCode([[0o7, 0o5]], [0o7]),
            5,
            None,
        ),
        (
            komm.ConvolutionalCode([[0b11, 0b10, 0b0], [0b0, 0b0, 0b1]]),
            3,
            None,
        ),
        (
            komm.ConvolutionalCode([[0o5, 0o7]]),
            6,
            [[1, 0], [1, 1]],
        ),
        (
            komm.ConvolutionalCode([[0o7, 0o5, 0o0], [0o0, 0o3, 0o2]]),
            4,
            [[1, 1], [1, 1], [1, 0]],
        ),
        (
            komm.HighRateConvolutionalCode([0o6, 0o7, 0o5, 0o1]),
            3,
            None,
        ),
        (  # [LC04, p. 602]
            komm.ConvolutionalCode([
                [3, 0, 1, 0, 3, 1, 1, 1],
                [0, 3, 1, 1, 2, 3, 1, 0],
                [2, 2, 3, 0, 0, 2, 3, 1],
                [0, 2, 0, 3, 2, 2, 2, 3],
            ]),
            3,
            None,
        ),
    ],
)
def test_viterbi_exhaustive_tail_biting(
    convolutional_code, num_blocks, puncturing_matrix, rng
):
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks, "tail-biting", puncturing_matrix
    )
    check_against_exhaustive(code, rng)


def test_viterbi_tail_biting_golay(rng):
    # [LC04, p. 602]
    convolutional_code = komm.ConvolutionalCode([
        [3, 0, 1, 0, 3, 1, 1, 1],
        [0, 3, 1, 1, 2, 3, 1, 0],
        [2, 2, 3, 0, 0, 2, 3, 1],
        [0, 2, 0, 3, 2, 2, 2, 3],
    ])
    code = komm.TerminatedConvolutionalCode(convolutional_code, 3, "tail-biting")
    golay = komm.GolayCode(extended=True)
    # Position i of the code is position perm[i] of the Golay code
    # fmt: off
    perm = [
        0, 1, 2, 3, 4, 6, 5, 19, 17, 20, 22, 11,
        12, 9, 23, 8, 14, 16, 21, 10, 7, 18, 13, 15,
    ]
    # fmt: on
    codewords = np.empty_like(code.codewords())
    codewords[:, perm] = code.codewords()
    assert sorted(map(tuple, codewords)) == sorted(map(tuple, golay.codewords()))
    viterbi = komm.ViterbiDecoder(code, input_type="soft")
    exhaustive = komm.ExhaustiveCodewordDecoder(golay, input_type="soft")
    r = rng.standard_normal((100, code.length))
    r_golay = np.empty_like(r)
    r_golay[:, perm] = r
    np.testing.assert_equal(
        exhaustive.decode_to_codeword(r_golay)[:, perm],
        viterbi.decode_to_codeword(r),
    )
