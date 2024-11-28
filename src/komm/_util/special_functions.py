from statistics import NormalDist

import numpy as np
import numpy.typing as npt

norm = NormalDist()


def _qfunc(x: float) -> float:
    return norm.cdf(-x)


def _qfuncinv(y: float) -> float:
    return -norm.inv_cdf(y) + 0.0  # + 0.0 to avoid -0.0


def qfunc(x: npt.ArrayLike) -> npt.NDArray[np.float64] | np.float64:
    r"""
    Computes the Gaussian Q-function. It is given by
    $$
        \mathrm{Q}(x) = \frac{1}{\sqrt{2\pi}} \int_x^\infty \mathrm{e}^{-u^2/2} \, \mathrm{d}u.
    $$

    Parameters:
        x (float | ArrayND[float]): The input to the function. May be any float or array of floats.

    Returns:
        y (SameAsInput): The value $y = \mathrm{Q}(x)$.

    Examples:
        >>> komm.qfunc(0.0)
        np.float64(0.5)

        >>> komm.qfunc([[-1.0], [0.0], [1.0]])
        array([[0.84134475],
               [0.5       ],
               [0.15865525]])
    """
    result = np.vectorize(_qfunc)(x)
    return np.float64(result) if np.isscalar(x) else result


def qfuncinv(y: npt.ArrayLike) -> npt.NDArray[np.float64] | np.float64:
    r"""
    Computes the inverse Gaussian Q-function.

    Parameters:
        y (float | ArrayND[float]): The input to the function. Should be a float or array of floats in the real interval $[0, 1]$.

    Returns:
        x (SameAsInput): The value $x = \mathrm{Q^{-1}}(y)$.

    Examples:
        >>> komm.qfuncinv(0.5)
        np.float64(0.0)

        >>> komm.qfuncinv([[0.841344746], [0.5], [0.158655254]])
        array([[-1.],
               [ 0.],
               [ 1.]])
    """
    result = np.vectorize(_qfuncinv)(y)
    return np.float64(result) if np.isscalar(y) else result


def logcosh(x: npt.ArrayLike) -> np.float64:
    # https://stackoverflow.com/a/57786270/
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)
