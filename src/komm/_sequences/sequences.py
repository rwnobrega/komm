import numpy as np
import numpy.typing as npt

from .._algebra.BinaryPolynomial import BinaryPolynomial


def barker_sequence(length: int) -> npt.NDArray[np.int_]:
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


def hadamard_matrix(length: int) -> npt.NDArray[np.int_]:
    h = np.array([[1]])
    g = np.array([[1, 1], [1, -1]])
    for _ in range(length.bit_length() - 1):
        h = np.kron(h, g).astype(int)
    return h


def lfsr_sequence(
    feedback_polynomial: BinaryPolynomial,
    start_state_polynomial: BinaryPolynomial,
) -> npt.NDArray[np.int_]:
    taps = (feedback_polynomial + BinaryPolynomial(1)).exponents()
    length = 2 ** taps[-1] - 1
    state = start_state_polynomial.coefficients(width=feedback_polynomial.degree)
    sequence = np.empty(length, dtype=int)
    for i in range(length):
        sequence[i] = state[-1]
        state[-1] = np.count_nonzero(state[taps - 1]) % 2
        state = np.roll(state, 1)
    return sequence


def default_primitive_polynomial(degree: int) -> BinaryPolynomial:
    if not 1 <= degree <= 16:
        raise ValueError("only degrees in the range [1 : 16] are implemented")
    return BinaryPolynomial(
        {
            1: 0b11,
            2: 0b111,
            3: 0b1011,
            4: 0b10011,
            5: 0b100101,
            6: 0b1000011,
            7: 0b10001001,
            8: 0b100011101,
            9: 0b1000010001,
            10: 0b10000001001,
            11: 0b100000000101,
            12: 0b1000001010011,
            13: 0b10000000011011,
            14: 0b100010001000011,
            15: 0b1000000000000011,
            16: 0b10000000010000011,
        }[degree]
    )
