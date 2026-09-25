from itertools import product

import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "convolutional_code, num_blocks, mode, snr, r, u_hat",
    [
        (  # Abrantes.10, p. 434--437.
            komm.ConvolutionalCode([[0b111, 0b101]]),
            4,
            "zero-termination",
            1.25,
            [-0.3, -0.1, +0.5, -0.2, -0.8, -0.5, +0.5, -0.3, -0.1, +0.7, -1.5, +0.4],
            [-1.78, -0.24, +1.97, -5.52],
        ),
        (  # Lin.Costello.04, p. 572--575.
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
def test_bcjr_exhaustive(convolutional_code, num_blocks, mode, rng):
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks, mode)
    decoder = komm.BCJRDecoder(code)
    li = rng.standard_normal((100, code.length))
    # Bitwise MAP over all messages
    u = np.array(list(product([0, 1], repeat=code.dimension)))
    metrics = 0.5 * li @ ((-1) ** code.encode(u)).T
    lo = [
        np.logaddexp.reduce(metrics[:, u[:, j] == 0], axis=-1)
        - np.logaddexp.reduce(metrics[:, u[:, j] == 1], axis=-1)
        for j in range(code.dimension)
    ]
    np.testing.assert_allclose(decoder.decode(li), np.stack(lo, axis=-1), atol=1e-8)


def test_bcjr_reliable_input(rng):
    code = komm.TerminatedConvolutionalCode(komm.ConvolutionalCode([[0o7, 0o5]]), 10)
    decoder = komm.BCJRDecoder(code)
    u = rng.integers(0, 2, (5, code.dimension))
    lo = decoder.decode(1000.0 * (-1) ** code.encode(u))
    assert np.all(np.isfinite(lo))
    np.testing.assert_equal(lo < 0, u == 1)
