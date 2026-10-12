from functools import reduce

import numpy as np
import numpy.typing as npt


def random_pmf(rng: np.random.Generator, size: int) -> npt.NDArray[np.floating]:
    pmf = rng.random(size)
    return pmf / pmf.sum()


def deterministic_pmf(size: int, index: int) -> npt.NDArray[np.floating]:
    pmf = np.zeros(size)
    pmf[index] = 1.0
    return pmf


def extended_pmf(pmf: npt.ArrayLike, order: int) -> npt.NDArray[np.floating]:
    return reduce(np.kron, [np.asarray(pmf)] * order)
