import numpy as np
import numpy.typing as npt


def branch_costs(lvalues: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    # Sum of |L| over disagreeing bits, for each output
    costs = np.zeros((*lvalues.shape[:-1], 1))
    for i in range(lvalues.shape[-1]):
        x = lvalues[..., i : i + 1]
        bit0, bit1 = costs + np.maximum(-x, 0), costs + np.maximum(x, 0)
        costs = np.concatenate([bit0, bit1], axis=-1)  # LSB first
    return costs


def marginalize(
    log_posteriors: npt.NDArray[np.floating],
    bits: npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:
    # L-value of each bit, in log domain
    metrics = log_posteriors[..., np.newaxis]
    l0 = np.logaddexp.reduce(np.where(bits == 0, metrics, -np.inf), axis=-2)
    l1 = np.logaddexp.reduce(np.where(bits == 1, metrics, -np.inf), axis=-2)
    return l0 - l1
