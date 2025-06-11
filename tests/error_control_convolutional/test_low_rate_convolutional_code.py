import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "degree, g_row, free_distance",
    [
        # Table 12.1(a): (4, 1) codes
        (1, [0o1, 0o1, 0o3, 0o3], 6),
        (2, [0o5, 0o5, 0o7, 0o7], 10),
        (3, [0o13, 0o13, 0o15, 0o17], 13),
        (4, [0o25, 0o27, 0o33, 0o37], 16),
        (5, [0o45, 0o53, 0o67, 0o77], 18),
        (6, [0o117, 0o127, 0o155, 0o171], 20),
        (7, [0o257, 0o311, 0o337, 0o355], 22),
        (8, [0o533, 0o575, 0o647, 0o711], 24),
        (9, [0o1173, 0o1325, 0o1467, 0o1751], 27),
        # Table 12.1(b): (3, 1) codes
        (1, [0o1, 0o3, 0o3], 5),
        (2, [0o5, 0o7, 0o7], 8),
        (3, [0o13, 0o15, 0o17], 10),
        (4, [0o25, 0o33, 0o37], 12),
        (5, [0o47, 0o53, 0o75], 13),
        (6, [0o117, 0o127, 0o155], 15),
        (7, [0o225, 0o331, 0o367], 16),
        (8, [0o575, 0o623, 0o727], 18),
        (9, [0o1167, 0o1375, 0o1545], 20),
        (10, [0o2325, 0o2731, 0o3747], 22),
        (11, [0o5745, 0o6471, 0o7553], 24),
        (12, [0o2371, 0o13725, 0o14733], 24),
        # Table 12.1(c): (2, 1) codes
        (1, [0o3, 0o1], 3),
        (2, [0o5, 0o7], 5),
        (3, [0o13, 0o17], 6),
        (4, [0o27, 0o31], 7),
        (5, [0o53, 0o75], 8),
        (6, [0o117, 0o155], 10),
        (7, [0o247, 0o371], 10),
        (8, [0o561, 0o753], 12),
        (9, [0o1131, 0o1537], 12),
        (10, [0o2473, 0o3217], 14),
        (11, [0o4325, 0o6747], 15),
        (12, [0o10627, 0o16765], 16),
        (13, [0o27251, 0o37363], 16),
    ],
)
def test_low_rate_convolutional_code_lin_costello(degree, g_row, free_distance):
    # [LC04, p. 539--540]
    code = komm.LowRateConvolutionalCode(g_row)
    n = len(g_row)
    assert code.num_input_bits == 1
    assert code.num_output_bits == n
    assert code.degree == degree
    assert code.is_catastrophic() is False
    assert code.free_distance() == free_distance

    # Test against general class
    feedforward_polynomials = [g_row]
    code1 = komm.ConvolutionalCode(feedforward_polynomials)
    G_mat, nus = code1.generator_matrix, code1.overall_constraint_length
    np.testing.assert_equal(code.generator_matrix, G_mat)
    np.testing.assert_equal(code.overall_constraint_length, nus)
    for _ in range(100):
        input = np.random.randint(0, 2, size=50)
        np.testing.assert_equal(code.encode(input), code1.encode(input))
