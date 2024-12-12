import numpy as np
import pytest

import komm


def test_convolutional_code_basic():
    # Lin.Costello.04, p. 454--456.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b1101, 0b1111]],
    )
    assert (code.num_output_bits, code.num_input_bits) == (2, 1)
    np.testing.assert_array_equal(code.constraint_lengths, [3])
    np.testing.assert_array_equal(code.memory_order, 3)
    np.testing.assert_array_equal(code.overall_constraint_length, 3)

    # Lin.Costello.04, p. 456--458.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]],
    )
    assert (code.num_output_bits, code.num_input_bits) == (3, 2)
    np.testing.assert_array_equal(code.constraint_lengths, [1, 1])
    np.testing.assert_array_equal(code.memory_order, 1)
    np.testing.assert_array_equal(code.overall_constraint_length, 2)

    # Ryan.Lin.09, p. 154.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
    )
    assert (code.num_output_bits, code.num_input_bits) == (2, 1)
    np.testing.assert_array_equal(code.constraint_lengths, [2])
    np.testing.assert_array_equal(code.memory_order, 2)
    np.testing.assert_array_equal(code.overall_constraint_length, 2)

    # Ibid.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
        feedback_polynomials=[0b111],
    )
    assert (code.num_output_bits, code.num_input_bits) == (2, 1)
    np.testing.assert_array_equal(code.constraint_lengths, [2])
    np.testing.assert_array_equal(code.memory_order, 2)
    np.testing.assert_array_equal(code.overall_constraint_length, 2)


def test_convolutional_code_state_space_representation():
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0o7, 0o5]])
    A_mat, B_mat, C_mat, D_mat = code.state_space_representation()
    np.testing.assert_array_equal(A_mat, [[0, 1], [0, 0]])
    np.testing.assert_array_equal(B_mat, [[1, 0]])
    np.testing.assert_array_equal(C_mat, [[1, 0], [1, 1]])
    np.testing.assert_array_equal(D_mat, [[1, 1]])

    # Heide Gluesing-Luerssen: On the Weight Distribution of Convolutional Codes, p. 9.
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b1111, 0b1101]])
    A_mat, B_mat, C_mat, D_mat = code.state_space_representation()
    np.testing.assert_array_equal(A_mat, [[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(B_mat, [[1, 0, 0]])
    np.testing.assert_array_equal(C_mat, [[1, 0], [1, 1], [1, 1]])
    np.testing.assert_array_equal(D_mat, [[1, 1]])


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials",
    [
        ([[0o7, 0o5]], None),
        ([[0o117, 0o155]], None),
        ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None),
        ([[0o27, 0o31]], [0o27]),
    ],
)
def test_convolutional_state_space_representation_2(
    feedforward_polynomials, feedback_polynomials
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    n, k, nu = code.num_output_bits, code.num_input_bits, code.overall_constraint_length

    A_mat, B_mat, C_mat, D_mat = code.state_space_representation()

    input_bits = np.random.randint(2, size=100 * k)
    output_bits = np.empty(n * input_bits.size // k, dtype=int)

    s = np.zeros(nu, dtype=int)

    for t, u in enumerate(np.reshape(input_bits, shape=(-1, k))):
        s, v = (s @ A_mat + u @ B_mat) % 2, (s @ C_mat + u @ D_mat) % 2
        output_bits[t * n : (t + 1) * n] = v

    encoder = komm.ConvolutionalStreamEncoder(code)
    np.testing.assert_array_equal(output_bits, encoder(input_bits))
