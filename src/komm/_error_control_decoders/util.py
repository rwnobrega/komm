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
    row_deg = A.sum(axis=1)
    b = np.array(b)
    while True:
        rows = np.flatnonzero(row_deg == 1)
        if rows.size == 0:
            return x, unknown, b
        cols = np.argmax(A[rows] & unknown, axis=1)
        x[cols] = b[rows]
        unknown[cols] = False
        cols = np.unique(cols)
        row_deg -= A[:, cols].sum(axis=1)
        b ^= A[:, cols[x[cols] == 1]].sum(axis=1) % 2
