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


def test_convolutional_code_encode_books():
    # Abrantes.10, p. 307.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
    )
    np.testing.assert_array_equal(
        code.encode([1, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    )

    # Lin.Costello.04, p. 454--456.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b1101, 0b1111]],
    )
    np.testing.assert_array_equal(
        code.encode([1, 0, 1, 1, 1, 0, 0, 0]),
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    )

    # Lin.Costello.04, p. 456--458.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]],
    )
    np.testing.assert_array_equal(
        code.encode([1, 1, 0, 1, 1, 0, 0, 0]),
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    )

    # Ryan.Lin.09, p. 154.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
    )
    np.testing.assert_array_equal(
        code.encode([1, 0, 0, 0]),
        [1, 1, 1, 0, 1, 1, 0, 0],
    )

    # Ibid.
    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b111, 0b101]],
        feedback_polynomials=[0b111],
    )
    np.testing.assert_array_equal(
        code.encode([1, 1, 1, 0]),
        [1, 1, 1, 0, 1, 1, 0, 0],
    )


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials, message, codeword",
    [
        # fmt: off
        (
            [[0o7, 0o5]],
            None,
            komm.int_to_bits(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int_to_bits(0xBE84A1FACDF49B0D258444495561C0D11F496CD12589847E89BDCE6CE5555B0039B0E5589B37E56CEBE5612BD2BDF7DC0000, 400),
        ),
        (
            [[0o117, 0o155]],
            None,
            komm.int_to_bits(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int_to_bits(0x3925A704C66355EB62F33DE3C4512D01A6D681376CCEC5F7FB8091BA4FF29B35456641CF63217AB7FD748A0560B5D4DC0000, 400),
        ),
        (
            [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]],
            None,
            komm.int_to_bits(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int_to_bits(0x6C889449F6801E93DAF4E498CCF75404897D7459CE571F1581A4D05B2011986C0C8501D4000, 300),
        ),
        (
            [[0o27, 0o31]],
            [0o27],
            komm.int_to_bits(0xCD698970BD55FE82A5E2BDD4DC8E3FF01C3F713E33EB2C9200, 200),
            komm.int_to_bits(0x525114C160C91F2AC5511933F5D6EA2ECEB9F48CC779F998D9D86A762D57DF2A23DAA7551F298D762D85D6E70E526B2C0000, 400),
        ),
        # fmt: on
    ],
)
def test_convolutional_code_encode_matlab(
    feedforward_polynomials, feedback_polynomials, message, codeword
):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    np.testing.assert_array_equal(code.encode(message), codeword)


def test_convolutional_code_state_space_representation():
    code = komm.ConvolutionalCode(feedforward_polynomials=[[0b111, 0b101]])
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

    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b11001, 0b10111, 0], [0, 0b1010, 0b1101]]
    )
    A_mat, B_mat, C_mat, D_mat = code.state_space_representation()
    np.testing.assert_array_equal(
        A_mat,
        [
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )
    np.testing.assert_array_equal(
        B_mat,
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
        ],
    )
    np.testing.assert_array_equal(
        C_mat,
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
        ],
    )
    np.testing.assert_array_equal(
        D_mat,
        [
            [1, 1, 0],
            [0, 0, 1],
        ],
    )

    code = komm.ConvolutionalCode(
        feedforward_polynomials=[[0b10111, 0b11001]],
        feedback_polynomials=[0b10111],
    )
    A_mat, B_mat, C_mat, D_mat = code.state_space_representation()
    np.testing.assert_array_equal(
        A_mat,
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
    )
    np.testing.assert_array_equal(
        B_mat,
        [
            [1, 0, 0, 0],
        ],
    )
    np.testing.assert_array_equal(
        C_mat,
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
        ],
    )
    np.testing.assert_array_equal(
        D_mat,
        [
            [1, 1],
        ],
    )


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials",
    [
        ([[0o7, 0o5]], None),
        ([[0o117, 0o155]], None),
        ([[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]], None),
        ([[0o27, 0o31]], [0o27]),
    ],
)
def test_convolutional_encoder_vs_fsm(feedforward_polynomials, feedback_polynomials):
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    n, k = code.num_output_bits, code.num_input_bits

    u = np.random.randint(2, size=100 * k)
    v1 = code.encode(u)

    fsm = code.finite_state_machine()
    input = komm.bits_to_int(u.reshape(-1, k))
    output, _ = fsm.process(input, 0)
    v2 = komm.int_to_bits(output, width=n).ravel()

    np.testing.assert_array_equal(v1, v2)


@pytest.mark.parametrize(
    "overall_constraint_length, feedforward_polynomials, free_distance",
    [
        # Table 12.1(a): (4, 1) codes
        (1, [[0o1, 0o1, 0o3, 0o3]], 6),
        (2, [[0o5, 0o5, 0o7, 0o7]], 10),
        (3, [[0o13, 0o13, 0o15, 0o17]], 13),
        (4, [[0o25, 0o27, 0o33, 0o37]], 16),
        (5, [[0o45, 0o53, 0o67, 0o77]], 18),
        (6, [[0o117, 0o127, 0o155, 0o171]], 20),
        (7, [[0o257, 0o311, 0o337, 0o355]], 22),
        (8, [[0o533, 0o575, 0o647, 0o711]], 24),
        (9, [[0o1173, 0o1325, 0o1467, 0o1751]], 27),
        # Table 12.1(b): (3, 1) codes
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
        # Table 12.1(c): (2, 1) codes
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
def test_convolutional_code_free_distance_g_lin_costello(
    overall_constraint_length, feedforward_polynomials, free_distance
):
    # [LC04, p. 539--540]
    code = komm.ConvolutionalCode(feedforward_polynomials)
    assert code.is_catastrophic() is False
    assert code.overall_constraint_length == overall_constraint_length
    assert code.free_distance() == free_distance


@pytest.mark.parametrize(
    "feedforward_polynomials, free_distance, forney_indices",
    [
        # Table 2: (2, 1) codes
        ([[0o1, 0o1]], 2, [0]),
        ([[0o1, 0o3]], 3, [1]),
        ([[0o5, 0o7]], 5, [2]),
        ([[0o13, 0o17]], 6, [3]),
        ([[0o23, 0o35]], 7, [4]),
        ([[0o53, 0o75]], 8, [5]),
        ([[0o133, 0o171]], 10, [6]),  # NASA
        ([[0o561, 0o753]], 12, [8]),
        ([[0o2335, 0o3661]], 14, [10]),
        # Table 3: (3, 1) codes
        ([[0o1, 0o1, 0o1]], 3, [0]),
        ([[0o1, 0o3, 0o3]], 5, [1]),
        ([[0o5, 0o7, 0o7]], 8, [2]),
        ([[0o13, 0o15, 0o17]], 10, [3]),
        ([[0o25, 0o33, 0o37]], 12, [4]),
        ([[0o47, 0o53, 0o75]], 13, [5]),
        ([[0o133, 0o145, 0o175]], 15, [6]),
        ([[0o225, 0o331, 0o367]], 16, [7]),
        ([[0o557, 0o663, 0o711]], 18, [8]),
        ([[0o1117, 0o1365, 0o1633]], 20, [9]),
        ([[0o2353, 0o2671, 0o3175]], 22, [10]),
        # Table 4: (3, 2) codes
        ([[0o1, 0o0, 0o1], [0o0, 0o1, 0o1]], 2, [0, 0]),
        ([[0o3, 0o2, 0o3], [0o2, 0o1, 0o1]], 3, [1, 1]),
        ([[0o1, 0o2, 0o3], [0o4, 0o1, 0o7]], 4, [1, 2]),
        ([[0o7, 0o4, 0o1], [0o2, 0o5, 0o7]], 5, [2, 2]),
        ([[0o3, 0o6, 0o7], [0o14, 0o1, 0o17]], 6, [2, 3]),
        ([[0o13, 0o6, 0o13], [0o6, 0o13, 0o17]], 7, [3, 3]),
        ([[0o3, 0o16, 0o15], [0o34, 0o31, 0o17]], 8, [3, 4]),  # Typo in the book
        ([[0o25, 0o30, 0o17], [0o50, 0o7, 0o65]], 9, [4, 5]),
        ([[0o63, 0o54, 0o31], [0o26, 0o53, 0o43]], 10, [5, 5]),
        # Table 5: (4, 1) codes
        ([[0o1, 0o1, 0o1, 0o1]], 4, [0]),
        ([[0o1, 0o3, 0o3, 0o3]], 7, [1]),
        ([[0o5, 0o7, 0o7, 0o7]], 10, [2]),
        ([[0o13, 0o15, 0o15, 0o17]], 13, [3]),
        ([[0o25, 0o27, 0o33, 0o37]], 16, [4]),
        ([[0o53, 0o67, 0o71, 0o75]], 18, [5]),
        ([[0o135, 0o135, 0o147, 0o163]], 20, [6]),
        ([[0o235, 0o275, 0o313, 0o357]], 22, [7]),
        ([[0o463, 0o535, 0o733, 0o745]], 24, [8]),
        ([[0o1117, 0o1365, 0o1633, 0o1653]], 27, [9]),
        ([[0o2327, 0o2353, 0o2671, 0o3175]], 29, [10]),  # Typo in the book
        # Table 6: (4, 3) codes
        (
            [[0o1, 0o0, 0o0, 0o1], [0o0, 0o1, 0o0, 0o1], [0o0, 0o0, 0o1, 0o1]],
            2,
            [0, 0, 0],
        ),
        (
            [[0o1, 0o1, 0o1, 0o1], [0o3, 0o1, 0o0, 0o0], [0o0, 0o3, 0o1, 0o0]],
            3,
            [0, 1, 1],
        ),
        (
            [[0o1, 0o1, 0o1, 0o1], [0o0, 0o3, 0o2, 0o1], [0o0, 0o2, 0o5, 0o5]],
            4,
            [0, 1, 2],
        ),
        (
            [[0o3, 0o2, 0o2, 0o3], [0o4, 0o3, 0o0, 0o7], [0o0, 0o2, 0o5, 0o5]],
            5,
            [1, 2, 2],
        ),
        (
            [[0o3, 0o4, 0o0, 0o7], [0o6, 0o1, 0o4, 0o3], [0o2, 0o6, 0o7, 0o1]],
            6,
            [2, 2, 2],
        ),
        (
            [[0o7, 0o6, 0o2, 0o1], [0o14, 0o5, 0o0, 0o15], [0o10, 0o4, 0o17, 0o1]],
            7,
            [2, 3, 3],
        ),
        (
            [[0o1, 0o14, 0o16, 0o3], [0o10, 0o13, 0o2, 0o7], [0o16, 0o0, 0o3, 0o13]],
            8,
            [3, 3, 3],
        ),
    ],
)
def test_convolutional_code_free_distance_g_mcelice(
    feedforward_polynomials, free_distance, forney_indices
):
    # [McE98, p. 1086--1088]
    code = komm.ConvolutionalCode(feedforward_polynomials)
    np.testing.assert_array_equal(code.constraint_lengths, forney_indices)
    assert code.is_catastrophic() is False
    assert code.free_distance() == free_distance


def test_convolutional_code_catastrophic_lin_costello():
    # [LC04, Example 11.9 (a)]
    code = komm.ConvolutionalCode([[0b1101, 0b1111]])
    assert code._minors_gcd() == komm.BinaryPolynomialFraction(0b1)
    assert code.is_catastrophic() is False
    # [LC04, Example 11.9 (b)]
    code = komm.ConvolutionalCode([[0b11, 0b10, 0b11], [0b10, 0b1, 0b1]])
    assert code._minors_gcd() == komm.BinaryPolynomialFraction(0b1)
    assert code.is_catastrophic() is False
    # [LC04, Example 11.10]
    code = komm.ConvolutionalCode([[0b11, 0b101]])
    assert code._minors_gcd() == komm.BinaryPolynomialFraction(0b11)
    assert code.is_catastrophic() is True


def test_convolutional_code_catastrophic_mceliece():
    # [McE98, Example 6.2]
    code = komm.ConvolutionalCode([[0b111, 0b101]])
    assert code.is_catastrophic() is False
    code = komm.ConvolutionalCode([[0b1001, 0b1111]])
    assert code.is_catastrophic() is True
    # [McE98, Example 6.4]
    code = komm.ConvolutionalCode([[0b1, 0b10, 0b1], [0b10, 0b11, 0b101]])
    assert code._minors_gcd() == komm.BinaryPolynomialFraction(0b111)
    assert code.is_catastrophic() is True
    # [McE98, Example 6.5]
    code = komm.ConvolutionalCode([[0b11, 0b10, 0b11], [0b10, 0b1, 0b11]])
    assert code._minors_gcd() == komm.BinaryPolynomialFraction(0b1)
    assert code.is_catastrophic() is False


@pytest.mark.parametrize(
    "feedforward_polynomials, feedback_polynomials, delta, is_catastrophic",
    [
        (
            [[0b1, 0b111, 0b101, 0b11], [0b10, 0b111, 0b100, 0b1]],
            [0b111, 0b10],
            0b111,
            True,
        ),
        (
            [[0b1, 0b111, 0b101, 0b11], [0b10, 0b111, 0b100, 0b1]],
            None,
            0b111,
            True,
        ),
        (
            [[0b1, 0b111, 0b101, 0b11], [0b0, 0b11, 0b10, 0b1]],
            None,
            0b1,
            False,
        ),
        (
            [[0b1, 0b10, 0b11, 0b0], [0b0, 0b11, 0b10, 0b1]],
            None,
            0b1,
            False,
        ),
        (
            [[0b11, 0b0, 0b1, 0b10], [0b10, 0b111, 0b100, 0b1]],
            None,
            0b111,
            True,
        ),
        (
            [[0b1, 0b1, 0b1, 0b1], [0b0, 0b11, 0b10, 0b1]],
            None,
            0b1,
            False,
        ),
        (
            [[0b11, 0b0, 0b1, 0b10], [0b1, 0b10, 0b11, 0b0]],
            None,
            0b10,
            False,
        ),
        (
            [[0b11, 0b0, 0b1, 0b10], [0b0, 0b11, 0b10, 0b1]],
            [0b11, 0b11],
            0b1,
            False,
        ),
    ],
)
def test_convolutional_code_mceliece_table_8(
    feedforward_polynomials, feedback_polynomials, delta, is_catastrophic
):
    # [McE98, Table 8, p. 1107; see p. 1079 for the matrices]
    code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    assert code._minors_gcd() == komm.BinaryPolynomialFraction(delta)
    assert code.is_catastrophic() == is_catastrophic
