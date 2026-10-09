import numpy as np

from ..types import Array1D


def zadoff_chu_sequence(length: int, root_index: int) -> Array1D[np.complexfloating]:
    n = np.arange(length)
    return np.exp(-1j * np.pi * root_index * n * (n + 1) / length)
