import numpy as np
import scipy as sp


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
    return 0.5 * sp.special.erfc(x / np.sqrt(2))

def qfuncinv(y):
    return np.sqrt(2) * sp.special.erfcinv(2 * y)

def entropy(pmf, base=2.0):
    base = np.exp(1) if base == 'e' else float(base)
    pmf = np.array(pmf, dtype=np.float)
    pmf_nonzero = pmf[pmf != 0.0]
    if not np.allclose(np.sum(pmf_nonzero), 1.0) or not np.alltrue(pmf_nonzero >= 0.0):
        raise ValueError("Invalid pmf")
    return -np.dot(pmf_nonzero, np.log(pmf_nonzero) / np.log(base))

def mutual_information(input_pmf, transition_probabilities, base=2.0):
    output_pmf = np.dot(input_pmf, transition_probabilities)
    entropy_output_prior = entropy(output_pmf, base)
    entropy_output_posterior = np.dot(input_pmf, np.apply_along_axis(entropy, 1, transition_probabilities))
    return entropy_output_prior - entropy_output_posterior
