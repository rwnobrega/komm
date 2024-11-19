import numpy as np


def rref(M):
    r"""
    Computes the row-reduced echelon form of the matrix M modulo 2.

    Loosely based on
    [1] https://gist.github.com/rgov/1499136
    """
    M_rref = np.copy(M)
    n_rows, n_cols = M_rref.shape

    def pivot(row):
        f_list = np.flatnonzero(row)
        if f_list.size > 0:
            return f_list[0]
        else:
            return n_rows

    for r in range(n_rows):
        # Choose the pivot.
        possible_pivots = [pivot(row) for row in M_rref[r:]]
        p = np.argmin(possible_pivots) + r

        # Swap rows.
        M_rref[[r, p]] = M_rref[[p, r]]

        # Pivot column.
        f = pivot(M_rref[r])
        if f >= n_cols:
            continue

        # Subtract the row from others.
        for i in range(n_rows):
            if i != r and M_rref[i, f] != 0:
                M_rref[i] = (M_rref[i] + M_rref[r]) % 2

    return M_rref


# TODO: this should be the main function!
#       rref should call this instead
def xrref(M):
    r"""
    Computes the row-reduced echelon form of the matrix M modulo 2.

    Returns
    =======
    P

    M_rref

    pivots

    Such that `M_rref = P @ M` (where `@` stands for matrix multiplication).
    """
    eye = np.eye(M.shape[0], dtype=int)

    augmented_M = np.concatenate((np.copy(M), np.copy(eye)), axis=1)
    augmented_M_rref = rref(augmented_M)

    M_rref = augmented_M_rref[:, : M.shape[1]]
    P = augmented_M_rref[:, M.shape[1] :]

    pivots = []
    j = 0
    while len(pivots) < M_rref.shape[0] and j < M_rref.shape[1]:
        if np.array_equal(M_rref[:, j], eye[len(pivots)]):
            pivots.append(j)
        j += 1

    return P, M_rref, np.array(pivots)


def right_inverse(M):
    P, _, s_indices = xrref(M)
    M_rref_ri = np.zeros(M.T.shape, dtype=int)

    M_rref_ri[s_indices] = np.eye(len(s_indices), M.shape[0])
    M_ri = np.dot(M_rref_ri, P) % 2
    return M_ri


def null_matrix(M):
    (k, n) = M.shape
    _, M_rref, s_indices = xrref(M)
    N = np.empty((n - k, n), dtype=int)
    p_indices = np.setdiff1d(np.arange(M.shape[1]), s_indices)
    N[:, p_indices] = np.eye(n - k, dtype=int)
    N[:, s_indices] = M_rref[:, p_indices].T
    return N
