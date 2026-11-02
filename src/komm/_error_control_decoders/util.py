import numpy as np
import numpy.typing as npt


def marginalize(
    log_posteriors: npt.NDArray[np.floating],
    bits: npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:
    # L-value of each bit, in log domain
    metrics = log_posteriors[..., np.newaxis]
    l0 = np.logaddexp.reduce(np.where(bits == 0, metrics, -np.inf), axis=-2)
    l1 = np.logaddexp.reduce(np.where(bits == 1, metrics, -np.inf), axis=-2)
    return l0 - l1
