import numpy as np
from tqdm import tqdm

from .._util.matrices import row_span
from ..types import Array2D


def extended_parity_submatrix(
    parity_submatrix: Array2D[np.integer],
) -> Array2D[np.integer]:
    last_column = (1 + np.sum(parity_submatrix, axis=1)) % 2
    extended_parity_submatrix = np.hstack([parity_submatrix, last_column[np.newaxis].T])
    return extended_parity_submatrix


def span_weight_distribution(matrix: Array2D[np.integer]) -> list[int]:
    n = matrix.shape[1]
    # Packed bits: faster XOR and popcount.
    packed = np.packbits(matrix, axis=1)
    # Batches of up to 2**10 words.
    low, high = row_span(packed[:10]), row_span(packed[10:])
    distribution = np.zeros(n + 1, dtype=int)
    desc = "Computing codeword weight distribution"
    for offset in tqdm(high, desc=desc, delay=2.5, unit_scale=len(low)):
        weights = np.bitwise_count(low ^ offset).sum(axis=1, dtype=int)
        distribution += np.bincount(weights, minlength=n + 1)
    return distribution.tolist()


def macwilliams_transform(distribution: list[int]) -> list[int]:
    # Weight distribution of the dual code.
    n, size = len(distribution) - 1, sum(distribution)
    dual = [0] * (n + 1)
    for i, a in enumerate(distribution):
        if a == 0:
            continue
        # Coefficients of (1 - z)^i (1 + z)^(n - i).
        prev, curr = 0, 1
        for j in range(n + 1):
            dual[j] += a * curr
            prev, curr = curr, ((n - 2 * i) * curr - (n - j + 1) * prev) // (j + 1)
    return [b // size for b in dual]
