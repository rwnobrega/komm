import itertools


import numpy as np

from scipy.special import erfc


def binlist2int(list_):
    return sum(1 << n for (n, b) in enumerate(list_) if b != 0)

def int2binlist(int_, width=None):
    return [int(b) for b in reversed(np.binary_repr(int_, width=width))]

def pack(list_, width):
    return np.apply_along_axis(binlist2int, 1, np.reshape(list_, newshape=(-1, width)))

def unpack(list_, width):
    return np.ravel(np.array([int2binlist(i, width=width) for i in list_]))

def binarray2hexstr(binarray):
    return hex(binlist2int(binarray))[:1:-1]

def hexstr2binarray(hexstr, width):
    intarray = np.array([int(x, 16) for x in hexstr], dtype=np.int)
    binstr = ''.join(['{:04b}'.format(x) for x in intarray])
    binarray = np.array([int(x) for x in binstr], dtype=np.int)
    return binarray[len(binarray)-width:]


def binary_iterator(shape):
    """
    [1] https://stackoverflow.com/a/30854608/3435475
    """
    size = np.prod(shape)
    shift = np.reshape(np.arange(size, dtype=np.int), newshape=shape)
    for j in range(2**size):
        yield j >> shift & 1


def binary_iterator_weight(n, w):
    """
    Generate all binary lists of length n and weight w.
    [1] http://stackoverflow.com/a/1851138/3435475
    """
    for bits in itertools.combinations(range(n), w):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        yield s


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


def entropy(pmf):
    pmf_nonzero = pmf[pmf != 0.0]
    return -np.dot(pmf_nonzero, np.log2(pmf_nonzero))


def mutual_information(input_pmf, transition_probabilities):
    output_pmf = np.dot(input_pmf, transition_probabilities)
    entropy_output_prior = entropy(output_pmf)
    entropy_output_posterior = np.dot(input_pmf, np.apply_along_axis(entropy, 1, transition_probabilities))
    return entropy_output_prior - entropy_output_posterior


def binomial(n, k):
    k = min(k, n - k)
    if k == 0:
        return 1
    if k < 0 or k > n:
        return 0
    else:
        return np.prod(range(n, n - k, -1)) // np.prod(range(1, k + 1))


def tag(**tags):
    """
    See PEP 232
    """
    def a(function):
        for key, value in tags.items():
            setattr(function, key, value)
        return function
    return a

def rst_table(table):
    lengths = [0] * len(table[0])
    for row in table:
        for i, entry in enumerate(row):
            lengths[i] = max(lengths[i], len(entry))

    border = '  '.join('=' * x for x in lengths)
    header, body = table[0], table[1:]

    rst = border + '\n    '
    header_str = '  '.join(entry.ljust(length) for length, entry in zip(lengths, header))

    rst += header_str + '\n    ' + border + '\n    '

    body_str = ''
    for row in body:
        body_str += '  '.join(entry.ljust(length) for length, entry in zip(lengths, row)) + '\n    '

    rst += body_str + border

    return rst
