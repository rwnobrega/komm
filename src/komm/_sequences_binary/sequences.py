import numpy as np

from .._algebra.BinaryPolynomial import BinaryPolynomial
from ..types import Array1D, Array2D


def barker_sequence(length: int) -> Array1D[np.integer]:
    return np.array(
        {
            2: [0, 1],
            3: [0, 0, 1],
            4: [0, 0, 1, 0],
            5: [0, 0, 0, 1, 0],
            7: [0, 0, 0, 1, 1, 0, 1],
            11: [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
            13: [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        }[length]
    )


def hadamard_matrix(length: int) -> Array2D[np.integer]:
    h = np.array([[1]])
    g = np.array([[1, 1], [1, -1]])
    for _ in range(length.bit_length() - 1):
        h = np.kron(h, g).astype(int)
    return h


def lfsr_sequence(
    feedback_polynomial: BinaryPolynomial,
    start_state_polynomial: BinaryPolynomial,
) -> Array1D[np.integer]:
    n = feedback_polynomial.degree
    taps = (feedback_polynomial + BinaryPolynomial(1)).exponents()
    length = 2**n - 1
    sequence = np.empty(length, dtype=int)
    sequence[:n] = start_state_polynomial.coefficients(width=n)[::-1]
    # Since p(X)^2 = p(X^2), tap positions can be doubled
    k, s = n, 1
    while k < length:
        if k >= 2 * n * s:
            s *= 2
        stop = min(k + taps[0] * s, length)
        sequence[k:stop] = sum(sequence[k - s * i : stop - s * i] for i in taps) % 2
        k = stop
    return sequence
