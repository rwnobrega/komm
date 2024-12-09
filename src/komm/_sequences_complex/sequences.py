import numpy as np
import numpy.typing as npt


def zadoff_chu_sequence(
    length: int, root_index: int
) -> npt.NDArray[np.complexfloating]:
    n = np.arange(length)
    return np.exp(-1j * np.pi * root_index * n * (n + 1) / length)
