import numpy as np
import pytest

import komm
from komm._util.matrices import (
    boolean_matmul,
    invariant_factors,
    matmul,
    matrix_power,
    pseudo_inverse,
    rank,
    rref,
    trellis_oriented_form,
    xrref,
)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        ((7,), (7, 3)),
        ((4, 7), (7, 3)),
        ((2, 3, 300), (300, 5)),
        ((4, 0), (0, 3)),
        ((4, 3), (3, 0)),
    ],
)
def test_matmul(x_shape, y_shape, rng):
    x = rng.integers(0, 2, x_shape)
    y = rng.integers(0, 2, y_shape)
    np.testing.assert_equal(matmul(x, y), x @ y % 2)
    np.testing.assert_equal(boolean_matmul(x, y), x @ y > 0)


@pytest.mark.parametrize("size", range(1, 6))
@pytest.mark.parametrize("exponent", [0, 1, 2, 3, 10, 101])
def test_matrix_power(size, exponent, rng):
    matrix = rng.integers(0, 2, (size, size))
    expected = np.eye(size, dtype=int)
    for _ in range(exponent):
        expected = expected @ matrix % 2
    np.testing.assert_equal(matrix_power(matrix, exponent), expected)


def test_matrix_power_invalid_exponent():
    with pytest.raises(ValueError):
        matrix_power([[1, 1], [1, 0]], -1)
    with pytest.raises(TypeError):
        matrix_power([[1, 1], [1, 0]], 2.0)  # type: ignore


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (
            [[1, 0], [1, 1]],
            [[1, 0], [0, 1]],
        ),
        (
            [[1, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ),
        (
            [[1, 1], [1, 1]],
            [[1, 1], [0, 0]],
        ),
        (
            [[1, 1, 0], [1, 0, 1], [0, 1, 1]],
            [[1, 0, 1], [0, 1, 1], [0, 0, 0]],
        ),
        (
            [[1, 0], [1, 1], [0, 1]],
            [[1, 0], [0, 1], [0, 0]],
        ),
        (
            [[1, 0, 1, 1], [0, 1, 1, 0]],
            [[1, 0, 1, 1], [0, 1, 1, 0]],
        ),
        (
            [[0, 1, 1, 0], [1, 0, 1, 1]],
            [[1, 0, 1, 1], [0, 1, 1, 0]],
        ),
        (
            [[0, 1, 1], [1, 1, 0], [1, 0, 1]],
            [[1, 0, 1], [0, 1, 1], [0, 0, 0]],
        ),
    ],
)
def test_rref_basic(matrix, expected):
    np.testing.assert_equal(rref(matrix), expected)


@pytest.mark.parametrize("size", range(1, 11))
def test_rref_zero_matrix(size):
    matrix = np.zeros((size, size), dtype=int)
    expected = np.zeros((size, size), dtype=int)
    np.testing.assert_equal(rref(matrix), expected)


@pytest.mark.parametrize("size", range(1, 11))
def test_rref_identity(size):
    matrix = np.eye(size, dtype=int)
    expected = np.eye(size, dtype=int)
    np.testing.assert_equal(rref(matrix), expected)


@pytest.mark.parametrize("size", range(1, 11))
def test_rref_properties(size, rng):
    for _ in range(10):
        matrix = rng.integers(0, 2, size=(size, size))
        result = rref(matrix)

        # Check that the result has the same shape
        assert result.shape == matrix.shape

        # Check that the result is binary
        assert np.all(np.logical_or(result == 0, result == 1))

        # Leading coefficient of a nonzero row is 1
        for i in range(result.shape[0]):
            leading_ones = np.flatnonzero(result[i] == 1)
            if leading_ones.size > 0:
                lead_col = leading_ones[0]
                assert result[i, lead_col] == 1

        # All entries above and below a leading 1 are 0
        for i in range(result.shape[0]):
            leading_ones = np.flatnonzero(result[i] == 1)
            if leading_ones.size > 0:
                lead_col = leading_ones[0]
                assert np.all(result[:i, lead_col] == 0)  # above
                assert np.all(result[i + 1 :, lead_col] == 0)  # below

        # The leading 1 in each row is to the right of the leading 1 in the previous row
        for i in range(1, result.shape[0]):
            leading_ones = np.flatnonzero(result[i] == 1)
            prev_leading_ones = np.flatnonzero(result[i - 1] == 1)
            if leading_ones.size > 0 and prev_leading_ones.size > 0:
                assert leading_ones[0] > prev_leading_ones[0]

        # The rank is maintained
        assert rank(matrix) == rank(result)


@pytest.mark.parametrize("n_rows", range(1, 6))
@pytest.mark.parametrize("n_cols", range(1, 6))
def test_xrref_random(n_rows, n_cols, rng):
    for _ in range(100):
        matrix = rng.integers(0, 2, size=(n_rows, n_cols))
        row_transform, reduced, _ = xrref(matrix)
        np.testing.assert_equal(np.dot(row_transform, matrix) % 2, reduced)


@pytest.mark.parametrize("n_rows", range(1, 6))
@pytest.mark.parametrize("n_cols", range(1, 6))
def test_pseudo_inverse_random(n_rows, n_cols, rng):
    for _ in range(100):
        matrix = rng.integers(0, 2, size=(n_rows, n_cols))
        p_inv = pseudo_inverse(matrix)
        assert p_inv.shape == (n_cols, n_rows)
        assert rank(matrix) == rank(p_inv)
        np.testing.assert_equal((matrix @ p_inv @ matrix) % 2, matrix)
        np.testing.assert_equal((p_inv @ matrix @ p_inv) % 2, p_inv)
        if rank(matrix) == n_rows:
            eye = np.eye(n_rows, dtype=int)
            np.testing.assert_equal((matrix @ p_inv) % 2, eye)
        if rank(matrix) == n_cols:
            eye = np.eye(n_cols, dtype=int)
            np.testing.assert_equal((p_inv @ matrix) % 2, eye)


def spans(matrix):
    n_cols = matrix.shape[1]
    leading = np.argmax(matrix, axis=1)
    trailing = n_cols - 1 - np.argmax(matrix[:, ::-1], axis=1)
    return leading, trailing


def state_profile(matrix):
    leading, trailing = spans(matrix)
    n_cols = matrix.shape[1]
    active = [(leading < t) & (trailing >= t) for t in range(n_cols + 1)]
    return [int(np.count_nonzero(a)) for a in active]


@pytest.mark.parametrize(
    "matrix, expected, profile",
    [
        (  # [LC04, Examples 9.1 and 9.4]
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
            [
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            [0, 1, 2, 3, 2, 3, 2, 1, 0],
        ),
        (  # [LC04, Example 9.10]
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            ],
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            ],
            [0, 1, 2, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 3, 2, 1, 0],
        ),
        (  # [MacK03, Sec. 25.4]
            [
                [1, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 1, 1, 0],
                [0, 0, 1, 0, 1, 1, 1],
                [0, 0, 0, 1, 0, 1, 1],
            ],
            [
                [1, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 1],
            ],
            [0, 1, 2, 3, 3, 2, 1, 0],
        ),
    ],
)
def test_trellis_oriented_form_books(matrix, expected, profile):
    # The form is not unique: check row space and spans
    tof = trellis_oriented_form(matrix)
    expected = np.asarray(expected)
    assert rank(np.vstack([tof, expected])) == tof.shape[0]
    np.testing.assert_equal(spans(tof), spans(expected))
    assert state_profile(tof) == profile


@pytest.mark.parametrize("n_rows", range(1, 7))
@pytest.mark.parametrize("n_cols", range(1, 9))
def test_trellis_oriented_form_random(n_rows, n_cols, rng):
    for _ in range(20):
        matrix = rng.integers(0, 2, size=(n_rows, n_cols))
        tof = trellis_oriented_form(matrix)
        r = rank(matrix)
        assert tof.shape == (r, n_cols)
        assert rank(np.vstack([matrix, tof])) == r
        leading, trailing = spans(tof)
        assert np.unique(leading).size == r
        assert np.unique(trailing).size == r
        # Active rows give the minimal state profile
        for t, active in enumerate(state_profile(tof)):
            assert active == rank(matrix[:, :t]) + rank(matrix[:, t:]) - r


@pytest.mark.parametrize(
    "matrix, factors",
    [
        (  # [McE98, p. 1128–1129]
            [
                [0b1, 0b111, 0b101, 0b11],
                [0b10, 0b111, 0b100, 0b1],
            ],
            [0b1, 0b111],
        ),
        (  # [JZ15, p. 63–65]
            [
                [0b11, 0b10, 0b1],
                [0b100, 0b1, 0b111],
            ],
            [0b1, 0b1],
        ),
        (
            [
                [0b1, 0b0, 0b0],
                [0b0, 0b1, 0b0],
            ],
            [0b1, 0b1],
        ),
        (
            [
                [0b1001, 0b0, 0b0, 0b1010],
                [0b0, 0b1001, 0b0, 0b1101],
                [0b0, 0b0, 0b1001, 0b1011],
            ],
            [0b1, 0b1001, 0b1001],
        ),
    ],
)
def test_invariant_factors(matrix, factors):
    assert invariant_factors(matrix) == [komm.BinaryPolynomial(f) for f in factors]
