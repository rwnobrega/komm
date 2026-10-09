from re import escape

import numpy as np
import pytest

import komm
from komm._error_control_convolutional.TerminatedConvolutionalCode import (
    ZeroTermination,
)


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
    n, k, m = code.length, code.dimension, code.redundancy
    G, H = code.generator_matrix, code.check_matrix
    assert (n, k, m) == parameters
    np.testing.assert_equal(G, generator_matrix)
    np.testing.assert_equal(G @ H.T % 2, np.zeros((k, m), dtype=int))
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
    # [LC04, pp. 587–590]
    convolutional_code = komm.ConvolutionalCode(*convolutional_args)
    code = komm.TerminatedConvolutionalCode(convolutional_code, *termination_args)
    assert (code.length, code.dimension, code.redundancy) == parameters
    np.testing.assert_equal(code.generator_matrix, generator_matrix)
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
        message = komm.int_to_bits([message_int], width=code.dimension)
        assert isinstance(code.strategy, ZeroTermination)
        tail = message @ code.strategy._tail_projector % 2
        message_with_tail = np.concatenate([message, tail])
        input = komm.bits_to_int(message_with_tail, width=k)
        _, fs = fsm.process(input, 0)
        assert fs == 0


@pytest.mark.parametrize(
    "convolutional_args",
    [
        ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None),
        ([[0o7, 0o5]], [0o7]),
    ],
)
def test_terminated_convolutional_code_tail_biting(convolutional_args):
    convolutional_code = komm.ConvolutionalCode(*convolutional_args)
    code = komm.TerminatedConvolutionalCode(convolutional_code, 5, "tail-biting")
    # Assert that the final state is the initial one.
    for message_int in range(2**code.dimension):
        message = komm.int_to_bits([message_int], width=code.dimension)
        initial_state = code.strategy.initial_state(message)
        _, final_state = convolutional_code.encode_with_state(message, initial_state)
        np.testing.assert_equal(final_state, initial_state)


@pytest.mark.parametrize(
    "convolutional_args, num_blocks",
    [
        (([[0b11, 0b1]], [0b11]), 6),
        (([[0o7, 0o5]], [0o7]), 6),
        (([[0o27, 0o31]], [0o27]), 5),
    ],
)
def test_terminated_convolutional_code_tail_biting_singular(
    convolutional_args, num_blocks
):
    # Here A^h + I is singular
    convolutional_code = komm.ConvolutionalCode(*convolutional_args)
    with pytest.raises(ValueError, match="tail-biting is impossible for this code"):
        komm.TerminatedConvolutionalCode(convolutional_code, num_blocks, "tail-biting")


@pytest.mark.parametrize(
    "feedforward_polynomials",
    [
        [[0o1, 0o1]],  # μ = 0
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
        message = komm.int_to_bits([i], width=code.dimension)
        np.testing.assert_equal(code.encode(message), code2.encode(message))


def test_terminated_convolutional_golay():
    # [LC04, p. 602]
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
def test_terminated_convolutional_mappings(feedforward_polynomials, mode, rng):
    code = komm.TerminatedConvolutionalCode(
        komm.ConvolutionalCode(feedforward_polynomials), num_blocks=10, mode=mode
    )
    k, m = code.dimension, code.redundancy
    for _ in range(100):
        u = rng.integers(0, 2, (3, 4, k))
        v = code.encode(u)
        np.testing.assert_equal(
            code.inverse_encode(v),
            u,
        )
        np.testing.assert_equal(
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
    with pytest.raises(ValueError, match="one or more inputs in 'v' are not valid"):
        r[0] = 1
        code.inverse_encode(r)  # Incorrect


@pytest.mark.parametrize(
    "puncturing_matrix, rate",
    [
        ([[1, 0], [1, 1]], 2 / 3),
        ([[1, 0, 1], [1, 1, 0]], 3 / 4),
    ],
)
def test_terminated_convolutional_code_punctured_lin_costello(puncturing_matrix, rate):
    # [LC04, Example 12.10]
    convolutional_code = komm.ConvolutionalCode([[0o5, 0o7]])
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, 6, "direct-truncation", puncturing_matrix
    )
    assert code.rate == rate
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, 4, "zero-termination", puncturing_matrix
    )
    assert code.minimum_distance() == 3


def test_terminated_convolutional_code_puncturing_matrix_of_ones():
    convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
    code = komm.TerminatedConvolutionalCode(convolutional_code, 4)
    punctured = komm.TerminatedConvolutionalCode(
        convolutional_code, 4, puncturing_matrix=[[1, 1, 1], [1, 1, 1]]
    )
    assert punctured.length == code.length
    np.testing.assert_equal(punctured.generator_matrix, code.generator_matrix)


def test_terminated_convolutional_code_depuncture(rng):
    convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
    code = komm.TerminatedConvolutionalCode(
        convolutional_code, 4, puncturing_matrix=[[1, 1], [1, 0]]
    )
    n_u, positions = 12, [3, 7, 11]
    # Bits, with erasures at the deleted positions
    v = code.encode(rng.integers(0, 2, (3, 4, 2 * code.dimension)))
    r = code.depuncture(v, 2)
    assert r.shape == (3, 4, 2 * n_u)
    np.testing.assert_equal(r.reshape(3, 4, 2, n_u)[..., positions], 2)
    np.testing.assert_equal(
        np.delete(r.reshape(3, 4, 2, n_u), positions, axis=-1),
        v.reshape(3, 4, 2, -1),
    )
    # L-values, with null L-values at the deleted positions
    li = rng.standard_normal((3, 4, 2 * code.length))
    lo = code.depuncture(li, 0.0)
    assert lo.shape == (3, 4, 2 * n_u)
    np.testing.assert_equal(lo.reshape(3, 4, 2, n_u)[..., positions], 0.0)
    np.testing.assert_equal(
        np.delete(lo.reshape(3, 4, 2, n_u), positions, axis=-1),
        li.reshape(3, 4, 2, -1),
    )
    # Float fill value gives floats
    assert code.depuncture(v, 0.0).dtype == float


@pytest.mark.parametrize(
    "puncturing_matrix, message",
    [
        ([[1, 2], [1, 1]], "elements of 'puncturing_matrix' must be 0 or 1"),
        ([1, 1, 1, 0], "'puncturing_matrix' must be a 2D-array (got shape (4,))"),
        ([[1, 1]], "'puncturing_matrix' must have one row per output bit"),
        ([[0, 0], [0, 0]], "'puncturing_matrix' must keep at least one bit"),
        ([[], []], "'puncturing_matrix' must keep at least one bit"),
        ([[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]], "'puncturing_matrix' period must divide"),
    ],
)
def test_terminated_convolutional_code_puncturing_matrix_invalid(
    puncturing_matrix, message
):
    convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
    with pytest.raises(ValueError, match=escape(message)):
        komm.TerminatedConvolutionalCode(
            convolutional_code, 4, "zero-termination", puncturing_matrix
        )


def test_terminated_convolutional_code_puncturing_matrix_not_integer():
    convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
    with pytest.raises(TypeError, match="'puncturing_matrix' must contain only"):
        komm.TerminatedConvolutionalCode(
            convolutional_code, 4, "zero-termination", [[1.0, 1.0], [1.0, 0.0]]
        )


def test_terminated_convolutional_code_invalid_mode():
    code = komm.ConvolutionalCode([[0o7, 0o5]])
    with pytest.raises(ValueError, match="'mode' must be 'direct-truncation'"):
        komm.TerminatedConvolutionalCode(code, 5, mode="zero")  # type: ignore
