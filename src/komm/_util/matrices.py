import numpy as np


def cartesian_product(A, B):
    r"""
    Computes the Cartesian product of two vectors/matrices.
    See SA15, eq. (2.2), where it is called the 'ordered direct product' and uses a different convention.
    """
    rA, cA = A.shape
    rB, cB = B.shape
    C = np.empty((rA * rB, cA + cB), dtype=A.dtype)
    for i, rowA in enumerate(A):
        for j, rowB in enumerate(B):
            C[j * rA + i, :] = np.concatenate((rowA, rowB))

    return C
