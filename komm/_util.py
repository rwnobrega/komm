import numpy as np
from scipy import special

__all__ = [
    "binlist2int",
    "int2binlist",
    "pack",
    "unpack",
    "qfunc",
    "qfuncinv",
    "entropy",
]


# Functions beginning with underscore:
# - Should not be used by the end user.
# - Should be as fast as possible.
# - May have assumptions on the input and on the output
#   (e.g., may assume the input is a list, or a numpy array, etc.).
#
# Functions without underscore:
# - Are available to the end user.
# - Should work when the input is a list, a numpy array, etc.
# - Should check the input whenever possible.
# - Should return a numpy array (instead of a list) whenever possible.

# TODO: Rename binlist2int and int2binlist to something better.
# TODO: Vectorize those functions (e.g., axis=1).


def _binlist2int(binlist):
    return sum(1 << i for (i, b) in enumerate(binlist) if b != 0)


def binlist2int(binlist):
    r"""
    Converts a bit array to its integer representation.

    Parameters:

        binlist (:obj:`list` or 1D-array of :obj:`int`): A list or array of $0$'s and $1$'s whose `i`-th element stands for the coefficient of $2^i$ in the binary representation of the output integer.

    Returns:

        integer (:obj:`int`): The integer representation of the input bit array.

    Examples:

        >>> komm.binlist2int([0, 1, 0, 1, 1])
        26

        >>> komm.binlist2int([0, 1, 0, 1, 1, 0, 0, 0])
        26
    """
    return _binlist2int(binlist)


def _int2binlist(integer, width=None):
    if width is None:
        width = max(integer.bit_length(), 1)
    return [(integer >> i) & 1 for i in range(width)]


def int2binlist(integer, width=None):
    r"""
    Converts an integer to its bit array representation.

    Parameters:

        int_ (:obj:`int`): The input integer. May be any nonnegative integer.

        width (:obj:`int`, optional): If this parameter is specified, the output will be filled with zeros on the right so that its length will be the specified value.

    Returns:

        binlist (1D-array of :obj:`int`): An array of $0$'s and $1$'s whose `i`-th element stands for the coefficient of $2^i$ in the binary representation of the input integer.

    Examples:

        >>> komm.int2binlist(26)
        array([0, 1, 0, 1, 1])

        >>> komm.int2binlist(26, width=8)
        array([0, 1, 0, 1, 1, 0, 0, 0])
    """
    return np.array(_int2binlist(integer, width))


def _pack(list_, width):
    return np.apply_along_axis(_binlist2int, 1, np.reshape(list_, newshape=(-1, width)))


def pack(list_, width):
    r"""
    Packs a given integer array.
    """
    return _pack(list_, width)


def _unpack(list_, width):
    return np.ravel([_int2binlist(i, width=width) for i in list_])


def unpack(list_, width):
    r"""
    Unpacks a given bit array.
    """
    return _unpack(list_, width)


def _qfunc(x):
    return 0.5 * special.erfc(x / np.sqrt(2))


def qfunc(x):
    r"""
    Computes the Gaussian Q-function. It is given by

    .. math::
       \mathrm{Q}(x) = \frac{1}{\sqrt{2\pi}} \int_x^\infty \mathrm{e}^{-u^2/2} \, \mathrm{d}u.

    Parameters:

        x (:obj:`float` or array of :obj:`float`): The input to the function. May be any float or array of floats.

    Returns:

        y (same as input): The value $y = \mathrm{Q}(x)$.

    Examples:

        >>> komm.qfunc(0.0)
        0.5

        >>> komm.qfunc([-1.0, 0.0, 1.0])
        array([0.84134475, 0.5       , 0.15865525])
    """
    return _qfunc(x)


def _qfuncinv(y):
    return np.sqrt(2) * special.erfcinv(2 * y)


def qfuncinv(y):
    r"""
    Computes the inverse Gaussian Q-function.

    Parameters:

        y (:obj:`float` or array of :obj:`float`): The input to the function. Should be a float or array of floats in the real interval $[0, 1]$.

    Returns:

        x (same as input): The value $x = \mathrm{Q^{-1}}(y)$.

    Examples:

        >>> komm.qfuncinv(0.5)  #doctest: +SKIP
        0.0

        >>> komm.qfuncinv([0.841344746, 0.5, 0.158655254])  #doctest: +SKIP
        array([-1., 0.,  1.])
    """
    return _qfuncinv(np.array(y))


def _entropy_base_e(pmf):
    # Assumptions:
    # - pmf is a 1D numpy array.
    # - pmf is a valid pmf.
    return -np.dot(pmf, np.log(pmf, where=(pmf > 0)))


def _entropy_base_2(pmf):
    # Assumptions: Same as _entropy_base_e.
    return -np.dot(pmf, np.log2(pmf, where=(pmf > 0)))


def _entropy(pmf, base=2.0):
    # Assumptions: Same as _entropy_base_e.
    if base == "e":
        return _entropy_base_e(pmf)
    elif base == 2:
        return _entropy_base_2(pmf)
    else:
        return _entropy_base_e(pmf) / np.log(base)


def entropy(pmf, base=2.0):
    r"""
    Computes the entropy of a random variable with a given :term:`pmf`. Let $X$ be a random variable with :term:`pmf` $p_X$ and alphabet $\mathcal{X}$. Its entropy is given by

    .. math::
       \mathrm{H}(X) = \sum_{x \in \mathcal{X}} p_X(x) \log \frac{1}{p_X(x)},

    By default, the base of the logarithm is $2$, in which case the entropy is measured in bits. See :cite:`Cover.Thomas.06` (Ch. 2).

    Parameters:

        pmf (1D-array of :obj:`float`): The probability mass function $p_X$ of the random variable. It must be a valid :term:`pmf`, that is, all of its values must be non-negative and sum up to $1$.

        base (:obj:`float` or :obj:`str`, optional): The base of the logarithm to be used. It must be a positive float or the string `'e'`. The default value is `2.0`.

    Returns:

        entropy (:obj:`float`): The entropy $\mathrm{H}(X)$ of the random variable.

    Examples:

        >>> komm.entropy([1/4, 1/4, 1/4, 1/4])
        2.0

        >>> komm.entropy([1/3, 1/3, 1/3], base=3.0)  #doctest: +SKIP
        1.0

        >>> komm.entropy([1.0, 1.0])
        Traceback (most recent call last):
        ...
        ValueError: Invalid pmf
    """
    pmf = np.array(pmf, dtype=float)
    if not np.allclose(np.sum(pmf), 1.0) or not np.alltrue(pmf >= 0.0):
        raise ValueError("Invalid pmf")
    return _entropy(pmf, base)


def _mutual_information(input_pmf, transition_probabilities, base=2.0):
    output_pmf = np.dot(input_pmf, transition_probabilities)
    entropy_output_prior = _entropy(output_pmf, base=base)
    entropy_base = lambda pmf: _entropy(pmf, base=base)
    entropy_output_posterior = np.dot(input_pmf, np.apply_along_axis(entropy_base, 1, transition_probabilities))
    return entropy_output_prior - entropy_output_posterior
