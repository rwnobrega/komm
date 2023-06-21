import numpy as np
from scipy import stats

_qfunc = stats.norm.sf
_qfuncinv = stats.norm.isf


def qfunc(x):
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
        0.5

        >>> komm.qfunc([-1.0, 0.0, 1.0])
        array([0.84134475, 0.5       , 0.15865525])
    """
    return _qfunc(x)


def qfuncinv(y):
    r"""
    Computes the inverse Gaussian Q-function.

    Parameters:

        y (float | ArrayND[float]): The input to the function. Should be a float or array of floats in the real interval $[0, 1]$.

    Returns:

        x (SameAsInput): The value $x = \mathrm{Q^{-1}}(y)$.

    Examples:

        >>> komm.qfuncinv(0.5)
        0.0

        >>> komm.qfuncinv([0.841344746, 0.5, 0.158655254])
        array([-1.,  0.,  1.])
    """
    return _qfuncinv(np.array(y))
