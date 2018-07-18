import numpy as np
import scipy as sp

__all__ = ['binlist2int', 'int2binlist', 'pack', 'unpack',
           'qfunc', 'qfuncinv',  'entropy', 'mutual_information']


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
    Pack a given integer array.
    """
    return _pack(list_, width)


def _unpack(list_, width):
    return np.ravel([_int2binlist(i, width=width) for i in list_])

def unpack(list_, width):
    """
    Unpack a given bit array.
    """
    return _unpack(list_, width)


def _qfunc(x):
    return 0.5 * sp.special.erfc(x / np.sqrt(2))

def qfunc(x):
    """
    Computes the gaussian Q-function of the input.
    """
    return _qfunc(x)


def _qfuncinv(y):
    return np.sqrt(2) * sp.special.erfcinv(2 * y)

def qfuncinv(x):
    """
    Computes the inverse gaussian Q-function of the input.
    """
    return _qfuncinv(x)


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
    Computes the entropy of a given pmf.
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

def mutual_information(input_pmf, transition_probabilities, base=2.0):
    """
    Computes the mutual information between a given pmf and transition probabilities.
    """
    return _mutual_information(input_pmf, transition_probabilities, base)
