import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, mode, snr, r, u_hat",
    [
        (  # [Abr10, pp. 434–437]
            komm.ConvolutionalCode([[0b111, 0b101]]),
            4,
            "zero-termination",
            1.25,
            [-0.3, -0.1, +0.5, -0.2, -0.8, -0.5, +0.5, -0.3, -0.1, +0.7, -1.5, +0.4],
            [-1.78, -0.24, +1.97, -5.52],
        ),
        (  # [LC04, pp. 572–575]
            komm.ConvolutionalCode([[0b11, 0b1]], [0b11]),
            3,
            "zero-termination",
            0.25,
            [-0.8, -0.1, -1.0, +0.5, +1.8, -1.1, -1.6, +1.6],
            [-0.48, -0.62, +1.02],
        ),
    ],
)
def test_bcjr(convolutional_code, num_blocks, mode, snr, r, u_hat):
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks, mode)
    decoder = komm.BCJRDecoder(code)
    li = 4 * snr * np.array(r)
    assert np.allclose(decoder.decode(li), u_hat, atol=0.05)


@pytest.mark.parametrize(
    "feedforward_polynomials",
    [
        [[0o7, 0o5, 0o0], [0o0, 0o3, 0o2]],
        [[0b11, 0b10, 0b0], [0b0, 0b0, 0b1]],  # Parallel transitions
    ],
)
def test_bcjr_two_input_bits(feedforward_polynomials, rng):
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials)
    code = komm.TerminatedConvolutionalCode(convolutional_code, 6)
    decoder = komm.BCJRDecoder(code, output_type="hard")
    u = rng.integers(0, 2, (20, code.dimension))
    li = 10.0 * (-1) ** code.encode(u)
    np.testing.assert_equal(decoder.decode(li), u)


def check_against_exhaustive(code, rng):
    decoder = komm.BCJRDecoder(code)
    exhaustive = komm.ExhaustiveBitwiseDecoder(code)
    li = rng.standard_normal((100, code.length))
    np.testing.assert_allclose(
        decoder.decode(li),
        exhaustive.decode(li),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        decoder.decode_to_codeword(li),
        exhaustive.decode_to_codeword(li),
        atol=1e-8,
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
def test_bcjr_exhaustive(convolutional_code, num_blocks, puncturing_matrix, mode, rng):
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
def test_bcjr_exhaustive_tail_biting(
    convolutional_code, num_blocks, puncturing_matrix, rng
):
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks, "tail-biting", puncturing_matrix
    )
    check_against_exhaustive(code, rng)


def test_bcjr_reliable_input(rng):
    code = komm.TerminatedConvolutionalCode(komm.ConvolutionalCode([[0o7, 0o5]]), 10)
    decoder = komm.BCJRDecoder(code)
    u = rng.integers(0, 2, (5, code.dimension))
    lo = decoder.decode(1000.0 * (-1) ** code.encode(u))
    assert np.all(np.isfinite(lo))
    np.testing.assert_equal(lo < 0, u == 1)


def test_bcjr_infinite_input():
    # Two code bits known for sure
    code = komm.TerminatedConvolutionalCode(komm.ConvolutionalCode([[0o7, 0o5]]), 8)
    decoder = komm.BCJRDecoder(code)
    u = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    r = 4.0 * (-1.0) ** code.encode(u)
    r[[0, 3]] *= np.inf  # One of each sign
    lo = decoder.decode(r)
    assert not np.any(np.isnan(lo))
    np.testing.assert_equal(lo < 0, u == 1)


@pytest.mark.parametrize(
    "code",
    [
        komm.HammingCode(3),
        komm.HammingCode(3, extended=True),
        komm.GolayCode(),
        komm.BCHCode(4, 3),
        komm.SingleParityCheckCode(5),
        komm.RepetitionCode(5),
        komm.BlockCode(generator_matrix=[[1, 1, 0, 1, 1], [0, 1, 1, 1, 0]]),
    ],
)
def test_bcjr_exhaustive_block(code, rng):
    check_against_exhaustive(code, rng)


non_systematic_codes = [
    komm.ReedMullerCode(4, 1),
    komm.ReedMullerCode(4, 2),
    komm.CyclicCode(7, generator_polynomial=0b1011, systematic=False),
    komm.PolarCode(4, [0, 1, 2, 3, 4, 5, 6, 8, 9]),
]


@pytest.mark.parametrize("code", non_systematic_codes)
def test_bcjr_exhaustive_block_codeword(code, rng):
    decoder = komm.BCJRDecoder(code)
    exhaustive = komm.ExhaustiveBitwiseDecoder(code)
    li = rng.standard_normal((100, code.length))
    np.testing.assert_allclose(
        decoder.decode_to_codeword(li),
        exhaustive.decode_to_codeword(li),
        atol=1e-8,
    )


@pytest.mark.parametrize("code", non_systematic_codes)
def test_bcjr_non_systematic(code):
    decoder = komm.BCJRDecoder(code)
    with pytest.raises(NotImplementedError, match="need a systematic generator matrix"):
        decoder.decode(np.zeros(code.length))
