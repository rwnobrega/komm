import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "mode, parameters, generator_matrix",
    [
        (
            "zero-termination",
            (8, 3, 3),
            [
                [1, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1],
            ],
        ),
        (
            "direct-truncation",
            (6, 3, 2),
            [
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 1, 1],
            ],
        ),
        (
            "tail-biting",
            (6, 3, 3),
            [
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1],
                [0, 1, 0, 0, 1, 1],
            ],
        ),
    ],
)
def test_terminated_convolutional_code_1(mode, parameters, generator_matrix):
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1, 0b11]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode=mode)
    (n, k, d) = (code.length, code.dimension, code.minimum_distance)
    (G, H) = code.generator_matrix, code.check_matrix
    assert (n, k, d) == parameters
    assert np.array_equal(G, generator_matrix)
    assert np.array_equal(np.dot(G, H.T) % 2, np.zeros((k, n - k), dtype=int))


def test_terminated_convolutional_code_tail_biting_1():
    # Lin.Costello.04, p. 586--587.
    convolutional_code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]]
    )
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks=6, mode="tail-biting"
    )
    assert (code.length, code.dimension, code.minimum_distance) == (12, 6, 3)
    assert np.array_equal(
        code.generator_matrix[0, :],
        [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    )


def test_terminated_convolutional_code_tail_biting_2():
    # Lin.Costello.04, p. 587--590.
    convolutional_code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]], feedback_polynomials=[0b111]
    )
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks=5, mode="tail-biting"
    )
    assert (code.length, code.dimension, code.minimum_distance) == (10, 5, 3)
    gen_mat = [
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    ]
    assert np.array_equal(code.generator_matrix, gen_mat)


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials",
    [
        ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None),
        ([[0o7, 0o5]], [0o7]),
    ],
)
def test_terminated_convolutional_code_zero_termination(
    feedforward_polynomials, feedback_polynomials
):
    convolutional_code = komm.ConvolutionalCode(
        feedforward_polynomials, feedback_polynomials
    )
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks=5, mode="zero-termination"
    )
    for message_int in range(2**code.dimension):
        print(message_int, 2**code.dimension)
        message = komm.int2binlist(message_int, width=code.dimension)
        tail = np.dot(message, code.tail_projector) % 2
        input_sequence = komm.pack(
            np.concatenate([message, tail]), width=convolutional_code._num_input_bits
        )
        _, fs = convolutional_code._finite_state_machine.process(
            input_sequence, initial_state=0
        )
        assert fs == 0


@pytest.mark.parametrize(
    "feedforward_polynomials",
    [
        [[0o7, 0o5]],
        [[0o3, 0o2, 0o3], [0o2, 0o1, 0o1]],
        [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
    ],
)
@pytest.mark.parametrize(
    "mode", ["zero-termination", "direct-truncation", "tail-biting"]
)
def test_terminated_convolutional_code_encoders(mode, feedforward_polynomials):
    convolutional_code = komm.ConvolutionalCode(
        feedforward_polynomials=feedforward_polynomials
    )
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=5, mode=mode)
    for i in range(2**code.dimension):
        message = komm.int2binlist(i, width=code.dimension)
        enc_mapping_1 = code.enc_mapping
        enc_mapping_2 = lambda u: komm.BlockCode.enc_mapping(code, u)
        assert np.array_equal(enc_mapping_1(message), enc_mapping_2(message))


def test_terminated_convolutional_golay():
    # Lin.Costello.04, p. 602.
    feedforward_polynomials = [
        [3, 0, 1, 0, 3, 1, 1, 1],
        [0, 3, 1, 1, 2, 3, 1, 0],
        [2, 2, 3, 0, 0, 2, 3, 1],
        [0, 2, 0, 3, 2, 2, 2, 3],
    ]
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials)
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, num_blocks=3, mode="tail-biting"
    )
    assert (code.length, code.dimension, code.minimum_distance) == (24, 12, 8)
