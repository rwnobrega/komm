import itertools
import numpy as np

def _extended_parity_submatrix(parity_submatrix):
    last_column = (1 + np.sum(parity_submatrix, axis=1)) % 2
    extended_parity_submatrix = np.hstack([parity_submatrix, last_column[np.newaxis].T])
    return extended_parity_submatrix


def _hamming_parity_submatrix(m):
    parity_submatrix = np.zeros((2**m - m - 1, m), dtype=int)
    i = 0
    for w in range(2, m + 1):
        for idx in itertools.combinations(range(m), w):
            parity_submatrix[i, list(idx)] = 1
            i += 1
    return parity_submatrix
