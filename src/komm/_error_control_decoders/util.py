from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def get_pbar(total: int, algorithm: str) -> "tqdm[Any]":
    return tqdm(
        total=total,
        desc=f"Decoding with {algorithm} algorithm",
        unit="block",
        delay=2.5,
    )


def marginalize(
    log_posteriors: npt.NDArray[np.floating],
    bits: npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:
    # L-value of each bit, in log domain
    metrics = log_posteriors[..., np.newaxis]
    l0 = np.logaddexp.reduce(np.where(bits == 0, metrics, -np.inf), axis=-2)
    l1 = np.logaddexp.reduce(np.where(bits == 1, metrics, -np.inf), axis=-2)
    return l0 - l1
