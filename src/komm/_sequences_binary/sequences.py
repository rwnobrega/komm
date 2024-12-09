import numpy as np
import numpy.typing as npt

from .._algebra.BinaryPolynomial import BinaryPolynomial


def barker_sequence(length: int) -> npt.NDArray[np.integer]:
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


def hadamard_matrix(length: int) -> npt.NDArray[np.integer]:
    h = np.array([[1]])
    g = np.array([[1, 1], [1, -1]])
    for _ in range(length.bit_length() - 1):
        h = np.kron(h, g).astype(int)
    return h


def lfsr_sequence(
    feedback_polynomial: BinaryPolynomial,
    start_state_polynomial: BinaryPolynomial,
) -> npt.NDArray[np.integer]:
    taps = (feedback_polynomial + BinaryPolynomial(1)).exponents()
    length = 2 ** taps[-1] - 1
    state = start_state_polynomial.coefficients(width=feedback_polynomial.degree)
    sequence = np.empty(length, dtype=int)
    for i in range(length):
        sequence[i] = state[-1]
        state[-1] = np.count_nonzero(state[taps - 1]) % 2
        state = np.roll(state, 1)
    return sequence
