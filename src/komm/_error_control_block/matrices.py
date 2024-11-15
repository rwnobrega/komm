import itertools as it

import numpy as np


def extended_parity_submatrix(parity_submatrix):
    last_column = (1 + np.sum(parity_submatrix, axis=1)) % 2
    extended_parity_submatrix = np.hstack([parity_submatrix, last_column[np.newaxis].T])
    return extended_parity_submatrix


def hamming_parity_submatrix(m, extended=False):
    parity_submatrix = np.zeros((2**m - m - 1, m), dtype=int)
    i = 0
    for w in range(2, m + 1):
        for idx in it.combinations(range(m), w):
            parity_submatrix[i, list(idx)] = 1
            i += 1
    if extended:
        parity_submatrix = extended_parity_submatrix(parity_submatrix)
    return parity_submatrix


def golay_parity_submatrix(extended=False):
    parity_submatrix = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        ]
    )
    if extended:
        parity_submatrix = extended_parity_submatrix(parity_submatrix)
    return parity_submatrix
