import numpy as np
import pytest

import komm
from komm._error_control_convolutional.terminations import ZeroTermination


@pytest.mark.parametrize(
    "mode, parameters, generator_matrix, min_distance",
    [
        (
            "direct-truncation",
            (6, 3, 3),
            [
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 1, 1],
            ],
            2,
        ),
        (
            "zero-termination",
            (8, 3, 5),
            [
                [1, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1],
            ],
            3,
        ),
        (
            "tail-biting",
            (6, 3, 3),
            [
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1],
                [0, 1, 0, 0, 1, 1],
            ],
            3,
        ),
    ],
)
def test_terminated_convolutional_code_parameters(
    mode, parameters, generator_matrix, min_distance
):
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1, 0b11]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode=mode)
    (n, k, m) = (code.length, code.dimension, code.redundancy)
    (G, H) = code.generator_matrix, code.check_matrix
    assert (n, k, m) == parameters
    np.testing.assert_array_equal(G, generator_matrix)
    np.testing.assert_array_equal(G @ H.T % 2, np.zeros((k, m), dtype=int))
    assert code.minimum_distance() == min_distance


@pytest.mark.parametrize(
    "convolutional_args, termination_args, parameters, generator_matrix, min_distance",
    [
        (
            ([[0b111, 0b101]], None),
            (6, "tail-biting"),
            (12, 6, 6),
            [
                [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            ],
            3,
        ),
        (
            ([[0b111, 0b101]], [0b111]),
            (5, "tail-biting"),
            (10, 5, 5),
            [
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            ],
            3,
        ),
    ],
)
def test_terminated_convolutional_code_tail_biting_lin_costello(
    convolutional_args, termination_args, parameters, generator_matrix, min_distance
):
    # Lin.Costello.04, p. 587--590.
    convolutional_code = komm.ConvolutionalCode(*convolutional_args)
    code = komm.TerminatedConvolutionalCode(convolutional_code, *termination_args)
    assert (code.length, code.dimension, code.redundancy) == parameters
    np.testing.assert_array_equal(code.generator_matrix, generator_matrix)
    assert code.minimum_distance() == min_distance


@pytest.mark.parametrize(
    "convolutional_args",
    [
        ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None),
        ([[0o7, 0o5]], [0o7]),
    ],
)
def test_terminated_convolutional_code_zero_termination(convolutional_args):
    convolutional_code = komm.ConvolutionalCode(*convolutional_args)
    k = convolutional_code.num_input_bits
    fsm = convolutional_code.finite_state_machine()
    code = komm.TerminatedConvolutionalCode(convolutional_code, 5, "zero-termination")
    # Assert that the final state is always 0.
    for message_int in range(2**code.dimension):
        message = komm.int_to_bits(message_int, width=code.dimension)
        assert isinstance(code._strategy, ZeroTermination)
        tail = message @ code._strategy._tail_projector % 2
        message_with_tail = np.concatenate([message, tail])
        input_sequence = komm.bits_to_int(message_with_tail.reshape(-1, k))
        _, fs = fsm.process(input_sequence, 0)
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
    "mode",
    ["zero-termination", "direct-truncation", "tail-biting"],
)
def test_terminated_convolutional_code_encoders(mode, feedforward_polynomials):
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials)
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=5, mode=mode)
    code2 = komm.BlockCode(generator_matrix=code.generator_matrix)
    for i in range(2**code.dimension):
        message = komm.int_to_bits(i, width=code.dimension)
        np.testing.assert_array_equal(code.encode(message), code2.encode(message))


def test_terminated_convolutional_golay():
    # Lin.Costello.04, p. 602.
    feedforward_polynomials = [
        [3, 0, 1, 0, 3, 1, 1, 1],
        [0, 3, 1, 1, 2, 3, 1, 0],
        [2, 2, 3, 0, 0, 2, 3, 1],
        [0, 2, 0, 3, 2, 2, 2, 3],
    ]
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials)
    code = komm.TerminatedConvolutionalCode(convolutional_code, 3, "tail-biting")
    assert (code.length, code.dimension, code.redundancy) == (24, 12, 12)
    assert code.minimum_distance() == 8


@pytest.mark.parametrize(
    "feedforward_polynomials",
    [[[0o7, 0o5]], [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]]],
)
@pytest.mark.parametrize(
    "mode",
    ["zero-termination", "direct-truncation", "tail-biting"],
)
def test_terminated_convolutional_mappings(feedforward_polynomials, mode):
    code = komm.TerminatedConvolutionalCode(
        komm.ConvolutionalCode(feedforward_polynomials), num_blocks=10, mode=mode
    )
    k, m = code.dimension, code.redundancy
    for _ in range(100):
        u = np.random.randint(0, 2, (3, 4, k))
        v = code.encode(u)
        np.testing.assert_array_equal(
            code.inverse_encode(v),
            u,
        )
        np.testing.assert_array_equal(
            code.check(v),
            np.zeros((3, 4, m)),
        )


@pytest.mark.parametrize(
    "mode", ["zero-termination", "direct-truncation", "tail-biting"]
)
def test_terminated_convolutional_unencode_invalid_input(mode):
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1, 0b11]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, num_blocks=3, mode=mode)
    r = np.zeros(code.length)
    code.inverse_encode(r)  # Correct
    with np.testing.assert_raises(ValueError):
        r[0] = 1
        code.inverse_encode(r)  # Incorrect
