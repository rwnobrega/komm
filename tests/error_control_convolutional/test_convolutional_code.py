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


@pytest.mark.parametrize(
    "overall_constraint_length, feedforward_polynomials, free_distance",
    [
        # (4, 1) convolutional codes
        (1, [[0o1, 0o1, 0o3, 0o3]], 6),
        (2, [[0o5, 0o5, 0o7, 0o7]], 10),
        (3, [[0o13, 0o13, 0o15, 0o17]], 13),
        (4, [[0o25, 0o27, 0o33, 0o37]], 16),
        (5, [[0o45, 0o53, 0o67, 0o77]], 18),
        (6, [[0o117, 0o127, 0o155, 0o171]], 20),
        (7, [[0o257, 0o311, 0o337, 0o355]], 22),
        (8, [[0o533, 0o575, 0o647, 0o711]], 24),
        (9, [[0o1173, 0o1325, 0o1467, 0o1751]], 27),
        # (3, 1) convolutional codes
        (1, [[0o1, 0o3, 0o3]], 5),
        (2, [[0o5, 0o7, 0o7]], 8),
        (3, [[0o13, 0o15, 0o17]], 10),
        (4, [[0o25, 0o33, 0o37]], 12),
        (5, [[0o47, 0o53, 0o75]], 13),
        (6, [[0o117, 0o127, 0o155]], 15),
        (7, [[0o225, 0o331, 0o367]], 16),
        (8, [[0o575, 0o623, 0o727]], 18),
        (9, [[0o1167, 0o1375, 0o1545]], 20),
        (10, [[0o2325, 0o2731, 0o3747]], 22),
        (11, [[0o5745, 0o6471, 0o7553]], 24),
        (12, [[0o2371, 0o13725, 0o14733]], 24),
        # (2, 1) convolutional codes
        (1, [[0o3, 0o1]], 3),
        (2, [[0o5, 0o7]], 5),
        (3, [[0o13, 0o17]], 6),
        (4, [[0o27, 0o31]], 7),
        (5, [[0o53, 0o75]], 8),
        (6, [[0o117, 0o155]], 10),
        (7, [[0o247, 0o371]], 10),
        (8, [[0o561, 0o753]], 12),
        (9, [[0o1131, 0o1537]], 12),
        (10, [[0o2473, 0o3217]], 14),
        (11, [[0o4325, 0o6747]], 15),
        (12, [[0o10627, 0o16765]], 16),
        (13, [[0o27251, 0o37363]], 16),
    ],
)
def test_convolutional_free_distance_g(
    overall_constraint_length, feedforward_polynomials, free_distance
):
    # Lin.Costello.04, p. 539--540
    convolutional_code = komm.ConvolutionalCode(feedforward_polynomials)
    assert convolutional_code.overall_constraint_length == overall_constraint_length
    code = komm.TerminatedConvolutionalCode(
        convolutional_code=convolutional_code,
        num_blocks=overall_constraint_length,
        mode="zero-termination",
    )
    assert code.minimum_distance() == free_distance
