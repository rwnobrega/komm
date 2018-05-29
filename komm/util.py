import numpy as np

from scipy.special import erfc, erfcinv


def binlist2int(list_):
    return sum(1 << i for (i, b) in enumerate(list_) if b != 0)

def int2binlist(int_, width=None):
    if width is None:
        width = max(int_.bit_length(), 1)
    return [(int_ >> i) & 1 for i in range(width)]

def pack(list_, width):
    return np.apply_along_axis(binlist2int, 1, np.reshape(list_, newshape=(-1, width)))

def unpack(list_, width):
    return np.ravel(np.array([int2binlist(i, width=width) for i in list_]))

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
