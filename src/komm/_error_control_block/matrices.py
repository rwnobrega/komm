from functools import reduce
from itertools import combinations

import numpy as np
import numpy.typing as npt

BinaryMatrix = npt.NDArray[np.int_]


def extended_parity_submatrix(parity_submatrix: BinaryMatrix) -> BinaryMatrix:
    last_column = (1 + np.sum(parity_submatrix, axis=1)) % 2
    extended_parity_submatrix = np.hstack([parity_submatrix, last_column[np.newaxis].T])
    return extended_parity_submatrix


def hamming_parity_submatrix(m: int, extended: bool = False) -> BinaryMatrix:
    parity_submatrix = np.zeros((2**m - m - 1, m), dtype=int)
    i = 0
    for w in range(2, m + 1):
        for idx in combinations(range(m), w):
            parity_submatrix[i, list(idx)] = 1
            i += 1
    if extended:
        parity_submatrix = extended_parity_submatrix(parity_submatrix)
    return parity_submatrix


def golay_parity_submatrix(extended: bool = False) -> BinaryMatrix:
    parity_submatrix = np.array([
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
    ])
    if extended:
        parity_submatrix = extended_parity_submatrix(parity_submatrix)
    return parity_submatrix


def reed_muller_generator_matrix(rho: int, mu: int) -> BinaryMatrix:
    # See [LC04, p. 105–114]. Assumes 0 <= rho < mu.
    v = np.empty((mu, 2**mu), dtype=int)
    for i in range(mu):
        block = np.hstack((
            np.zeros(2 ** (mu - i - 1), dtype=int),
            np.ones(2 ** (mu - i - 1), dtype=int),
        ))
        v[mu - i - 1] = np.tile(block, 2**i)

    G_list: list[npt.NDArray[np.int_]] = []
    for ell in range(rho, 0, -1):
        for indices in combinations(range(mu), ell):
            row = reduce(np.multiply, v[indices, :])
            G_list.append(row)
    G_list.append(np.ones(2**mu, dtype=int))

    return np.array(G_list, dtype=int)


def cordaro_wagner_generator_matrix(n: int) -> BinaryMatrix:
    # See [CW67].
    d = int(np.ceil(2 * n / 3)) - 1
    q = (n + 1) // 3
    return np.hstack((
        np.repeat([[1], [0]], repeats=d - q, axis=1),
        np.repeat([[0], [1]], repeats=q, axis=1),
        np.repeat([[1], [1]], repeats=n - d, axis=1),
    ))
