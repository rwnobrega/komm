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


def peel(
    A: npt.NDArray[np.bool_],
    b: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool_], npt.NDArray[np.integer]]:
    x = np.zeros(A.shape[1], dtype=int)
    unknown = np.ones(A.shape[1], dtype=bool)
    while True:
        rows = np.flatnonzero(A[:, unknown].sum(axis=1) == 1)
        if rows.size == 0:
            return x, unknown, b
        for i in rows:
            cols = np.flatnonzero(A[i] & unknown)
            if cols.size != 1:  # Already solved in this pass.
                continue
            j = cols[0]
            x[j] = b[i]
            unknown[j] = False
            if x[j]:
                b = b ^ A[:, j]
