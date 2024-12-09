import numpy as np
import numpy.typing as npt
from tqdm import tqdm

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
