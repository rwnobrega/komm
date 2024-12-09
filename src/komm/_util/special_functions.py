from math import gamma
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

norm = NormalDist()

gamma = np.vectorize(gamma)


def _gaussian_q(x: float) -> float:
    return norm.cdf(-x)


def gaussian_q(x: npt.ArrayLike) -> npt.NDArray[np.floating] | np.floating:
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
    return result[()] if np.ndim(result) == 0 else result


def _gaussian_q_inv(y: float) -> float:
    if y == 0:
        return np.inf
    if y == 1:
        return -np.inf
    return -norm.inv_cdf(y)


def gaussian_q_inv(y: npt.ArrayLike) -> npt.NDArray[np.floating] | np.floating:
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
    return result[()] if np.ndim(result) == 0 else result


def _auxiliary_p(
    m: int, a: float, b: float, max_iter: int = 100, tol: float = 1e-16
) -> float:
    # Auxiliary function for the Marcum Q-function
    # See [https://en.wikipedia.org/wiki/Marcum_Q-function].
    ms = np.arange(m)
    result = np.sum(np.exp(-b) * (b**ms) / gamma(ms + 1))
    for i in range(m, max_iter):
        ks = np.arange(i - m + 1)
        inner_sum = np.sum(np.exp(-m * a) * ((m * a) ** ks) / gamma(ks + 1))
        term = np.exp(-b) * (b**i) / gamma(i + 1) * (1 - inner_sum)
        result += term
        if abs(term) < tol:
            break
    return result


def marcum_q(
    m: int, a: npt.ArrayLike, x: npt.ArrayLike
) -> npt.NDArray[np.floating] | np.floating:
    r"""
    Computes the Marcum Q-function. It is given by
    $$
        \mathrm{Q}_m(a; x) = \int_x^\infty u \left( \frac{u}{a} \right)^{m-1}  I\_{m-1}(a x) \exp \left( -\frac{u^2 + a^2}{2} \right) \mathrm{d}u,
    $$
    where $I\_{m-1}$ is the modified Bessel function of the first kind. This corresponds to the complementary cumulative distribution function of the non-central chi distribution with $2m$ degrees of freedom and non-centrality parameter $a$. For more details, see [Wikipedia: Marcum Q-function](https://en.wikipedia.org/wiki/Marcum_Q-function).

    Parameters:
        m: The order of the Marcum Q-function. Should be a positive integer.
        a: The value of $a$. Should be a float or array of floats.
        x: The input to the function. Should be a float or array of floats.

    Returns:
        y: The value $y = \mathrm{Q}_m(a; x)$.

    Examples:
        >>> komm.marcum_q(1, 1, 1)
        np.float64(0.7328798037968204)

        >>> komm.marcum_q(2, 0.5, [1.2, 1.4, 1.6])
        array([0.85225816, 0.76472056, 0.66139663])
    """
    a = np.asarray(a)
    x = np.asarray(x)
    if not m > 0:
        raise ValueError("'m' must be positive")
    if not np.all(a >= 0) or not np.all(x >= 0):
        raise ValueError("'a' and 'x' must be non-negative")
    result = np.vectorize(_auxiliary_p)(m, a**2 / (2 * m), x**2 / 2)
    return result[()] if np.ndim(result) == 0 else result


def logcosh(x: npt.ArrayLike) -> np.floating:
    # https://stackoverflow.com/a/57786270/
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)
