import numpy as np


def gcd(x, y, ring):
    """
    Performs the `Euclidean algorithm<https://en.wikipedia.org/wiki/Euclidean_algorithm>`_ with :code:`x` and :code:`y`.
    """
    if y == ring(0):
        return x
    else:
        return gcd(y, x % y, ring)


def xgcd(x, y, ring):
    """
    Performs the `extended Euclidean algorithm<https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm>`_ with :code:`x` and :code:`y`.
    """
    if x == ring(0):
        return y, ring(0), ring(1)
    else:
        d, s, t = xgcd(y % x, x, ring)
        return d, t - s * (y // x), s


def power(x, n, ring):
    """
    Returns :code:`x**n` using the `exponentiation by squaring<https://en.wikipedia.org/wiki/Exponentiation_by_squaring>`_ algorithm.
    """
    if n == 0:
        return ring(1)
    elif n == 1:
        return x
    elif n % 2 == 0:
        return power(x * x, n // 2, ring)
    else:
        return x * power(x * x, n // 2, ring)


def binary_horner(poly, x):
    """
    Returns the binary polynomial :code:`poly` evaluated at point :code:`x`, using `Horner's method <https://en.wikipedia.org/wiki/Horner's_method>`_.  Any Python object supporting the operations of addition, subtraction, and multiplication may serve as the input point.
    """
    result = x - x  # zero
    for coefficient in reversed(poly.coefficients()):
        result *= x
        if coefficient:
            result += coefficient
    return result


def horner(poly, x):
    """
    Returns the polynomial :code:`poly` evaluated at point :code:`x`, using `Horner's method <https://en.wikipedia.org/wiki/Horner's_method>`_.  Any Python object supporting the operations of addition, subtraction, and multiplication may serve as the input point.
    """
    result = x - x  # zero
    for coefficient in reversed(poly.coefficients()):
        result = result * x + coefficient
    return result


def rref(M):
    """
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
    """
    Computes the row-reduced echelon form of the matrix M modulo 2.

    Returns
    =======
    P

    M_rref

    pivots

    Such that :obj:`M_rref = P @ M` (where :obj:`@` stands for matrix multiplication).
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
