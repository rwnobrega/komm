import numpy as np
import scipy as sp

__all__ = ['binlist2int', 'int2binlist', 'pack', 'unpack',
           'qfunc', 'qfuncinv',  'entropy']


# Functions beggining with underscore:
# - Should not be used by the end user.
# - Should be as fast as possible.
# - May have assumptions on the input and on the output
#   (e.g., may assume the input is a list, or a numpy array, etc.).
#
# Functions without underscore:
# - Are available to the end-user.
# - Should work when the input is a list, a numpy array, etc.
# - Should check the input whenever possible.
# - Should return a numpy array (intead of a list) whenever possible.

# TODO: Rename binlist2int and int2binlist to something better.
# TODO: Vectorize those functions (e.g., axis=1).


def _binlist2int(list_):
    return sum(1 << i for (i, b) in enumerate(list_) if b != 0)

def binlist2int(list_):
    """
    Converts a bit array to its integer representation.
    """
    return _binlist2int(list_)


def _int2binlist(int_, width=None):
    if width is None:
        width = max(int_.bit_length(), 1)
    return [(int_ >> i) & 1 for i in range(width)]

def int2binlist(int_, width=None):
    """
    Converts an integer to its bit array representation.
    """
    return np.array(_int2binlist(int_, width))


def _pack(list_, width):
    return np.apply_along_axis(_binlist2int, 1, np.reshape(list_, newshape=(-1, width)))

def pack(list_, width):
    """
    Packs a given integer array.
    """
    return _pack(list_, width)


def _unpack(list_, width):
    return np.ravel([_int2binlist(i, width=width) for i in list_])

def unpack(list_, width):
    """
    Unpacks a given bit array.
    """
    return _unpack(list_, width)


def _qfunc(x):
    return 0.5 * sp.special.erfc(x / np.sqrt(2))

def qfunc(x):
    """
    Computes the gaussian Q-function. It is given by

    .. math::

       \\mathrm{Q}(x) = \\frac{1}{\\sqrt{2\\pi}} \\int_x^\\infty \\mathrm{e}^{-u^2/2} \\, \\mathrm{d}u.

    **Input:**

    :code:`x` : :obj:`float` or array of :obj:`float`
        The input to the function. May be any float or array of floats.

    **Output:**

    :code:`y` : same as input
        The value :math:`y = \\mathrm{Q}(x)`.

    .. rubric:: Examples

    >>> komm.qfunc(0.0)
    0.5

    >>> komm.qfunc([-1.0, 0.0, 1.0])
    array([0.84134475, 0.5       , 0.15865525])
    """
    return _qfunc(x)


def _qfuncinv(y):
    return np.sqrt(2) * sp.special.erfcinv(2 * y)

def qfuncinv(x):
    """
    Computes the inverse gaussian Q-function.

    **Input:**

    :code:`y` : :obj:`float` or array of :obj:`float`
        The input to the function. Should be a float or array of floats in the real interval :math:`[0, 1]`.

    **Output:**

    :code:`x` : same as input
        The value :math:`x = \\mathrm{Q^{-1}}(y)`.

    >>> komm.qfuncinv(0.5)  #doctest:+SKIP
    0.0

    >>> komm.qfuncinv([0.841344746, 0.5, 0.158655254])  #doctest:+SKIP
    array([-1., 0.,  1.])
    """
    return _qfuncinv(np.array(x))


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
    if base == 'e':
        return _entropy_base_e(pmf)
    elif base == 2:
        return _entropy_base_2(pmf)
    else:
        return _entropy_base_e(pmf) / np.log(base)

def entropy(pmf, base=2.0):
    """
    Computes the entropy of a random variable with a given :term:`pmf`. Let :math:`X` be a random variable with :term:`pmf` :math:`p_X` and alphabet :math:`\\mathcal{X}`. Its entropy is given by

    .. math::

       \\mathrm{H}(X) = \\sum_{x \\in \\mathcal{X}} p_X(x) \\log \\frac{1}{p_X(x)},

    By default, the base of the logarithm is :math:`2`, in which case the entropy is measured in bits.

    References: :cite:`Cover.Thomas.06` (Ch. 2)

    **Input:**

    :code:`pmf` : 1D-array of :obj:`float`
        The probability mass function :math:`p_X` of the random variable. It must be a valid :term:`pmf`, that is, all of its values must be non-negative and sum up to :math:`1`.

    :code:`base` : :obj:`float` or :obj:`str`, optional
        The base of the logarithm to be used. It must be a positive float or the string :code:`'e'`. The default value is :code:`2.0`.

    **Output:**

    :code:`entropy` : :obj:`float`
        The entropy :math:`\\mathrm{H}(X)` of the random variable.

    .. rubric:: Examples

    >>> komm.entropy([1/4, 1/4, 1/4, 1/4])
    2.0

    >>> komm.entropy([1/3, 1/3, 1/3], base=3.0)  #doctest:+SKIP
    1.0

    >>> komm.entropy([1.0, 1.0])
    Traceback (most recent call last):
     ...
    ValueError: Invalid pmf
    """
    pmf = np.array(pmf, dtype=np.float)
    if not np.allclose(np.sum(pmf), 1.0) or not np.alltrue(pmf >= 0.0):
        raise ValueError("Invalid pmf")
    return _entropy(pmf, base)


def _mutual_information(input_pmf, transition_probabilities, base=2.0):
    output_pmf = np.dot(input_pmf, transition_probabilities)
    entropy_output_prior = _entropy(output_pmf, base)
    entropy_output_posterior = np.dot(input_pmf, np.apply_along_axis(_entropy, 1, transition_probabilities))
    return entropy_output_prior - entropy_output_posterior
