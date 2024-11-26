import numpy as np
import numpy.typing as npt


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
