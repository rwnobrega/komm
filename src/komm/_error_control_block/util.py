import numpy as np
import numpy.typing as npt


def extended_parity_submatrix(
    parity_submatrix: npt.NDArray[np.integer],
) -> npt.NDArray[np.integer]:
    last_column = (1 + np.sum(parity_submatrix, axis=1)) % 2
    extended_parity_submatrix = np.hstack([parity_submatrix, last_column[np.newaxis].T])
    return extended_parity_submatrix
