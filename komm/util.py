import numpy as np

from scipy.special import erfc, erfcinv


def binlist2int(list_):
    return sum(1 << n for (n, b) in enumerate(list_) if b != 0)

def int2binlist(int_, width=None):
    return [int(b) for b in reversed(np.binary_repr(int_, width=width))]

def pack(list_, width):
    return np.apply_along_axis(binlist2int, 1, np.reshape(list_, newshape=(-1, width)))

def unpack(list_, width):
    return np.ravel(np.array([int2binlist(i, width=width) for i in list_]))


#http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
POPCOUNT_TABLE16 = [0] * 2**16
for index in range(len(POPCOUNT_TABLE16)):
    POPCOUNT_TABLE16[index] = (index & 1) + POPCOUNT_TABLE16[index >> 1]

def hamming_distance_16(a, b):
    v = a ^ b
    return (POPCOUNT_TABLE16[ v        & 0xffff] +
            POPCOUNT_TABLE16[(v >> 16) & 0xffff])


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

def qfuncinv(y):
    return np.sqrt(2) * erfcinv(2 * y)


def entropy(pmf):
    pmf_nonzero = pmf[pmf != 0.0]
    return -np.dot(pmf_nonzero, np.log2(pmf_nonzero))

def mutual_information(input_pmf, transition_probabilities):
    output_pmf = np.dot(input_pmf, transition_probabilities)
    entropy_output_prior = entropy(output_pmf)
    entropy_output_posterior = np.dot(input_pmf, np.apply_along_axis(entropy, 1, transition_probabilities))
    return entropy_output_prior - entropy_output_posterior


def tag(**tags):
    """
    See PEP 232
    """
    def a(function):
        for key, value in tags.items():
            setattr(function, key, value)
        return function
    return a
