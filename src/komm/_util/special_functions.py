from statistics import NormalDist

import numpy as np
import numpy.typing as npt

norm = NormalDist()


def _gaussian_q(x: float) -> float:
    return norm.cdf(-x)


def gaussian_q(x: npt.ArrayLike) -> npt.NDArray[np.float64] | np.float64:
    r"""
    Computes the Gaussian Q-function. It is given by
    $$
        \mathrm{Q}(x) = \frac{1}{\sqrt{2\pi}} \int_x^\infty \mathrm{e}^{-u^2/2} \, \mathrm{d}u.
    $$
    This corresponds to the complementary cumulative distribution function of the standard gaussian distribution. For more details, see [Wikipedia: Q-function](https://en.wikipedia.org/wiki/Q-function).

    Parameters:
        x: The input to the function. Should be a float or array of floats.

    Returns:
        y: The value $y = \mathrm{Q}(x)$.

    Examples:
        >>> komm.gaussian_q(0.0)
        np.float64(0.5)

        >>> komm.gaussian_q([[-1.0], [0.0], [1.0]])
        array([[0.84134475],
               [0.5       ],
               [0.15865525]])
    """
    result = np.vectorize(_gaussian_q)(x)
    return np.float64(result) if np.ndim(result) == 0 else result


def _gaussian_q_inv(y: float) -> float:
    if y == 0:
        return np.inf
    if y == 1:
        return -np.inf
    return -norm.inv_cdf(y)


def gaussian_q_inv(y: npt.ArrayLike) -> npt.NDArray[np.float64] | np.float64:
    r"""
    Computes the inverse Gaussian Q-function.

    Parameters:
        y: The input to the function. Should be a float or array of floats in the real interval $[0, 1]$.

    Returns:
        x: The value $x = \mathrm{Q^{-1}}(y)$.

    Examples:
        >>> komm.gaussian_q_inv(0.5)
        np.float64(0.0)

        >>> komm.gaussian_q_inv([[0.841344746], [0.5], [0.158655254]])
        array([[-1.],
               [ 0.],
               [ 1.]])
    """
    result = np.vectorize(_gaussian_q_inv)(y) + 0.0  # + 0.0 to avoid -0.0
    return np.float64(result) if np.ndim(result) == 0 else result


def logcosh(x: npt.ArrayLike) -> np.float64:
    # https://stackoverflow.com/a/57786270/
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)
