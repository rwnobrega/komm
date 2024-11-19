from math import e

import numpy as np
import pytest

from komm._util.matrices import pseudo_inverse, rank, rref, xrref


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
    np.testing.assert_array_equal(rref(matrix), expected)


@pytest.mark.parametrize("size", range(1, 11))
def test_rref_zero_matrix(size):
    matrix = np.zeros((size, size), dtype=int)
    expected = np.zeros((size, size), dtype=int)
    np.testing.assert_array_equal(rref(matrix), expected)


@pytest.mark.parametrize("size", range(1, 11))
def test_rref_identity(size):
    matrix = np.eye(size, dtype=int)
    expected = np.eye(size, dtype=int)
    np.testing.assert_array_equal(rref(matrix), expected)


@pytest.mark.parametrize("size", range(1, 11))
def test_rref_properties(size):
    for _ in range(10):
        matrix = np.random.randint(0, 2, size=(size, size))
        result = rref(matrix)

        # Check that the result has the same shape
        assert result.shape == matrix.shape

        # Check that the result is binary
        assert np.all(np.logical_or(result == 0, result == 1))

        # Leading coefficient of a nonzero row is 1
        nonzero_rows = np.any(result != 0, axis=1)
        for row in result[nonzero_rows]:
            first_nonzero = np.nonzero(row)[0]
            if first_nonzero.size > 0:
                assert row[first_nonzero[0]] == 1

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
def test_xrref_random(n_rows, n_cols):
    for _ in range(100):
        matrix = np.random.randint(0, 2, size=(n_rows, n_cols))
        row_transform, reduced, _ = xrref(matrix)
        np.testing.assert_array_equal(np.dot(row_transform, matrix) % 2, reduced)


@pytest.mark.parametrize("n_rows", range(1, 6))
@pytest.mark.parametrize("n_cols", range(1, 6))
def test_pseudo_inverse_random(n_rows, n_cols):
    for _ in range(100):
        matrix = np.random.randint(0, 2, size=(n_rows, n_cols))
        p_inv = pseudo_inverse(matrix)
        if rank(matrix) == n_rows:
            eye = np.eye(n_rows, dtype=int)
            np.testing.assert_array_equal((matrix @ p_inv) % 2, eye)
        else:
            np.testing.assert_array_equal((matrix @ p_inv @ matrix) % 2, matrix)
            np.testing.assert_array_equal((p_inv @ matrix @ p_inv) % 2, p_inv)
