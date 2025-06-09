import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "degree, h_row, free_distance",
    [
        # Table 12.1(d): (3, 2) codes
        (2, [0o7, 0o5, 0o3], 3),
        (3, [0o13, 0o15, 0o17], 4),
        (4, [0o27, 0o31, 0o23], 5),
        (5, [0o73, 0o57, 0o71], 6),
        (6, [0o121, 0o147, 0o123], 7),
        (7, [0o241, 0o227, 0o313], 8),
        (8, [0o477, 0o631, 0o555], 8),
        (9, [0o1327, 0o1423, 0o1051], 9),
        (10, [0o3013, 0o2137, 0o2621], 10),
        # Table 12.1(e): (4, 3) codes
        (2, [0o6, 0o7, 0o5, 0o1], 3),
        (3, [0o12, 0o15, 0o13, 0o11], 4),
        (4, [0o31, 0o37, 0o25, 0o33], 4),
        (5, [0o75, 0o57, 0o73, 0o47], 5),
        (6, [0o141, 0o133, 0o135, 0o107], 6),
        (7, [0o267, 0o315, 0o341, 0o211], 6),
        (8, [0o661, 0o733, 0o757, 0o535], 7),
        (9, [0o1371, 0o1157, 0o1723, 0o1475], 8),
    ],
)
def test_high_rate_convolutional_code_lin_costello(degree, h_row, free_distance):
    # [LC04, p. 540]
    # Note that the book displays the columns in reverse order.
    code = komm.HighRateConvolutionalCode(h_row)
    n = len(h_row)
    assert code.num_input_bits == n - 1
    assert code.num_output_bits == n
    assert code.degree == degree
    assert code.is_catastrophic() is False
    assert code.free_distance() == free_distance

    # Test against general class
    feedforward_polynomials = np.zeros((n - 1, n), dtype=int)
    feedback_polynomials = np.zeros(n - 1, dtype=int)
    for i in range(n - 1):
        feedforward_polynomials[i, i] = h_row[-1]
        feedforward_polynomials[i, n - 1] = h_row[i]
        feedback_polynomials[i] = h_row[-1]
    code1 = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
    np.testing.assert_equal(code.generator_matrix(), code1.generator_matrix())
    for _ in range(100):
        input = np.random.randint(0, 2, size=50 * (n - 1))
        np.testing.assert_equal(code.encode(input), code1.encode(input))
