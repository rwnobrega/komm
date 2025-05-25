from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from komm._algebra import BinaryPolynomial

ArrayInt = npt.NDArray[np.integer]


def _pivot(row: ArrayInt) -> int:
    f_list = np.flatnonzero(row)
    return f_list[0] if f_list.size > 0 else row.size


def rref(matrix: npt.ArrayLike) -> ArrayInt:
    r"""
    Computes the row-reduced echelon form of a matrix in $\ZZ_2$.

    Parameters:
        matrix: The matrix to reduce. Its elements must be `0` or `1`.

    Returns:
        reduced: The row-reduced echelon form of the matrix.

    Examples:
        >>> rref([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
        array([[1, 1, 0],
               [0, 0, 1],
               [0, 0, 0]])

    References:
        [1] https://gist.github.com/rgov/1499136
    """
    reduced = np.asarray(matrix, dtype=int)
    n_rows, n_cols = reduced.shape
    for r in tqdm(range(n_rows), desc="Row-reducing matrix", unit="row", delay=2.5):
        # Choose the pivot
        possible_pivots = [_pivot(row) for row in reduced[r:]]
        p = np.argmin(possible_pivots) + r
        # Swap rows
        reduced[r], reduced[p] = reduced[p].copy(), reduced[r].copy()
        # Pivot column
        f = _pivot(reduced[r])
        if f >= n_cols:
            continue
        # Subtract the row from others
        for i in range(n_rows):
            if i != r and reduced[i, f] != 0:
                reduced[i] = (reduced[i] + reduced[r]) % 2
    return reduced


def xrref(matrix: npt.ArrayLike) -> tuple[ArrayInt, ArrayInt, ArrayInt]:
    r"""
    Computes the row-reduced echelon form of a matrix in $\ZZ_2$. Returns the row transformation matrix, the reduced matrix, and the pivot indices. The relation between the input matrix `matrix` and the reduced matrix `reduced` is given by the equation `reduced = row_transform @ matrix`.

    Parameters:
        matrix: The matrix to reduce. Its elements must be `0` or `1`.

    Returns:
        row_transform: The row transformation matrix.
        reduced: The row-reduced echelon form of the matrix.
        pivots: The pivot indices.

    Examples:
        >>> matrix = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
        >>> row_transform, reduced, pivots = xrref(matrix)
        >>> row_transform
        array([[0, 1, 0],
               [0, 0, 1],
               [1, 1, 1]])
        >>> reduced
        array([[1, 1, 0],
               [0, 0, 1],
               [0, 0, 0]])
        >>> pivots
        array([0, 2])
        >>> np.array_equal((row_transform @ matrix) % 2, reduced)
        True
    """
    matrix = np.asarray(matrix, dtype=int)
    n_rows, n_cols = matrix.shape
    eye = np.eye(n_rows, dtype=int)
    augmented = np.concatenate((matrix, eye), axis=1)
    augmented_rref = rref(augmented)

    reduced = augmented_rref[:, :n_cols]
    row_transform = augmented_rref[:, n_cols:]

    pivots: list[int] = []
    for j, col in enumerate(reduced.T):
        if len(pivots) >= n_rows:
            break
        if np.array_equal(col, eye[len(pivots)]):
            pivots.append(j)

    return row_transform, reduced, np.array(pivots)


def rank(matrix: npt.ArrayLike) -> int:
    r"""
    Computes the rank of a matrix in $\ZZ_2$.

    Parameters:
        matrix: The matrix. Its elements must be `0` or `1`.

    Returns:
        rank: The rank of the matrix.

    Examples:
        >>> rank([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
        2
    """
    _, _, pivots = xrref(matrix)
    return len(pivots)


def pseudo_inverse(matrix: npt.ArrayLike) -> ArrayInt:
    r"""
    Computes a pseudo inverse of a matrix in $\ZZ_2$.

    Parameters:
        matrix: The matrix. Its elements must be `0` or `1`.

    Returns:
        p_inverse: A pseudo inverse of the matrix.

    Examples:
        >>> matrix = np.array([[1, 0, 1], [1, 1, 1]])
        >>> p_inverse = pseudo_inverse(matrix)
        >>> p_inverse
        array([[1, 0],
               [1, 1],
               [0, 0]])
        >>> (matrix @ p_inverse) % 2
        array([[1, 0],
               [0, 1]])

        >>> matrix = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
        >>> p_inverse = pseudo_inverse(matrix)
        >>> p_inverse
        array([[0, 1, 0],
               [0, 0, 0],
               [0, 0, 1]])
        >>> np.array_equal((matrix @ p_inverse @ matrix) % 2, matrix)
        True
        >>> np.array_equal((p_inverse @ matrix @ p_inverse) % 2, p_inverse)
        True
    """
    row_transform, reduced, pivots = xrref(matrix)
    reduced_inverse = np.zeros_like(reduced.T)
    if pivots.size == 0:
        return reduced_inverse
    reduced_inverse[pivots] = np.eye(pivots.size, reduced.shape[0])
    p_inverse = np.dot(reduced_inverse, row_transform) % 2
    return p_inverse


def null_matrix(matrix: npt.ArrayLike) -> ArrayInt:
    r"""
    Computes a null matrix of a matrix in $\ZZ_2$.
    """
    _, reduced, pivots = xrref(matrix)
    n_rows, n_cols = reduced.shape
    null = np.empty((n_cols - n_rows, n_cols), dtype=int)
    not_pivots = np.setdiff1d(range(n_cols), pivots)
    null[:, not_pivots] = np.eye(n_cols - n_rows, dtype=int)
    null[:, pivots] = reduced[:, not_pivots].T
    return null


def block_diagonal(arrays: Sequence[npt.ArrayLike]) -> npt.NDArray[Any]:
    arrays_np = [np.asarray(a) for a in arrays]
    total_rows = sum(a.shape[0] for a in arrays_np)
    total_cols = sum(a.shape[1] for a in arrays_np)
    result = np.zeros((total_rows, total_cols), dtype=arrays_np[0].dtype)
    r_off, c_off = 0, 0
    for a in arrays_np:
        r, c = a.shape
        result[r_off : r_off + r, c_off : c_off + c] = a
        r_off += r
        c_off += c
    return result


def invariant_factors(matrix: npt.ArrayLike) -> list[BinaryPolynomial]:
    r"""
    Computes the invariant factors of a matrix over $\ZZ_2[D]$.

    Parameters:
        matrix: The matrix. Its elements must be [binary polynomials](/ref/BinaryPolynomial) or integers to be converted to the former.

    Returns:
        factors: The invariant factors of the matrix.

    Examples:
        >>> invariant_factors([[0b1, 0b111, 0b101, 0b11], [0b10, 0b111, 0b100, 0b1]])
        [BinaryPolynomial(0b1), BinaryPolynomial(0b111)]
    """
    # See [McE98, Appendix B, p. 1128]
    BP = BinaryPolynomial
    matrix = np.array(matrix, dtype=int)
    k, n = matrix.shape
    if k > n:
        raise ValueError("number of rows cannot exceed number of columns")
    if np.all(matrix == 0):
        return []

    # E1: Find entry of least size and move it to position (0,0)
    i0, j0, d0, a0 = -1, -1, np.inf, BP(0b0)
    for i, j in product(range(k), range(n)):
        a = BP(matrix[i, j])
        if a == 0b0:
            continue
        d = a.degree
        if d < d0:
            i0, j0, d0, a0 = i, j, d, a
    matrix[0, :], matrix[i0, :] = matrix[i0, :].copy(), matrix[0, :].copy()
    matrix[:, 0], matrix[:, j0] = matrix[:, j0].copy(), matrix[:, 0].copy()

    # E2a
    done = False
    while not done:
        done = True
        for j in range(1, n):
            a = BP(matrix[0, j])
            if a == 0:
                continue
            q, r = divmod(a, a0)
            if r != 0:
                a0 = r
                for i in range(k):
                    matrix[i, j] = BP(matrix[i, j]) - q * BP(matrix[i, 0])
                matrix[:, 0], matrix[:, j] = matrix[:, j].copy(), matrix[:, 0].copy()
                done = False
                break
    # E2b
    done = False
    while not done:
        done = True
        for i in range(1, k):
            a = BP(matrix[i, 0])
            if a == 0:
                continue
            q, r = divmod(a, a0)
            if r != 0:
                a0 = r
                for j in range(n):
                    matrix[i, j] = BP(matrix[i, j]) - q * BP(matrix[0, j])
                matrix[0, :], matrix[i, :] = matrix[i, :].copy(), matrix[0, :].copy()
                done = False
                break

    # E3a
    for j in range(1, n):
        q = BP(matrix[0, j]) // a0
        for i in range(k):
            matrix[i, j] = BP(matrix[i, j]) - q * BP(matrix[i, 0])
    # E3b
    for i in range(1, k):
        q = BP(matrix[i, 0]) // a0
        for j in range(n):
            matrix[i, j] = BP(matrix[i, j]) - q * BP(matrix[0, j])

    return [a0] + invariant_factors(matrix[1:, 1:])
