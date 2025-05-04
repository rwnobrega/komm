import numpy as np
import numpy.typing as npt


def rect(x: npt.ArrayLike) -> npt.NDArray[np.floating]:
    x = np.asarray(x)
    return 1.0 * (-0.5 <= x) * (x < 0.5)


def tri(x: npt.ArrayLike) -> npt.NDArray[np.floating]:
    x = np.asarray(x)
    return (1.0 - np.abs(x)) * (-1.0 <= x) * (x < 1.0) + 0.0


def raised_cosine(x: npt.ArrayLike, α: float) -> npt.NDArray[np.floating]:
    x = np.asarray(x)
    return np.pi / 4 * np.sinc(x) * (np.sinc(α * x + 0.5) + np.sinc(α * x - 0.5))
