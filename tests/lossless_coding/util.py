import numpy as np
import numpy.typing as npt


def random_pmf(size: int) -> npt.NDArray[np.floating]:
    pmf = np.random.rand(size)
    return pmf / pmf.sum()


def shuffle_pmf(pmf: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return pmf[np.random.permutation(pmf.size)]


def deterministic_pmf(size: int, index: int) -> npt.NDArray[np.floating]:
    pmf = np.zeros(size)
    pmf[index] = 1.0
    return pmf
