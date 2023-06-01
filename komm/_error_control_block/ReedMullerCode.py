import functools
import itertools

import numpy as np

from .BlockCode import BlockCode
from .._aux import tag

class ReedMullerCode(BlockCode):
    """
    Reed--Muller code. It is a linear block code (:obj:`BlockCode`) defined by two integers :math:`\\rho` and :math:`\\mu`, which must satisfy :math:`0 \\leq \\rho < \\mu`. See references for more details. The resulting code is denoted by :math:`\\mathrm{RM}(\\rho, \\mu)`, and has the following parameters:

    - Length: :math:`n = 2^{\\mu}`
    - Dimension: :math:`k = 1 + {\\mu \\choose 1} + \\cdots + {\\mu \\choose \\rho}`
    - Redundancy: :math:`m = 1 + {\\mu \\choose 1} + \\cdots + {\\mu \\choose \\mu - \\rho - 1}`
    - Minimum distance: :math:`d = 2^{\\mu - \\rho}`

    References: :cite:`Lin.Costello.04` (p. 105--114)

    .. rubric:: Decoding methods

    [[decoding_methods]]

    .. rubric:: Notes

    - For :math:`\\rho = 0` it reduces to a repetition code (:class:`RepetitionCode`).
    - For :math:`\\rho = 1` it reduces to a lengthened simplex code (:class:`SimplexCode`).
    - For :math:`\\rho = \\mu - 2` it reduces to an extended Hamming code (:class:`HammingCode`).
    - For :math:`\\rho = \\mu - 1` it reduces to a single parity check code (:class:`SingleParityCheckCode`).

    .. rubric:: Examples

    >>> code = komm.ReedMullerCode(1, 5)
    >>> (code.length, code.dimension, code.minimum_distance)
    (32, 6, 16)
    >>> code.generator_matrix  #doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
           [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    >>> code.encode([0, 0, 0, 0, 0, 1])  #doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> recvword = np.ones(32, dtype=int); recvword[[2, 10, 15, 16, 17, 19, 29]] = 0
    >>> code.decode(recvword)
    array([0, 0, 0, 0, 0, 1])
    """
    def __init__(self, rho, mu):
        """
        Constructor for the class. It expects the following parameters:

        :code:`rho` : :obj:`int`
            The parameter :math:`\\rho` of the code.

        :code:`mu` : :obj:`int`
            The parameter :math:`\\mu` of the code.

        The parameters must satisfy :math:`0 \\leq \\rho < \\mu`.
        """
        if not 0 <= rho < mu:
            raise ValueError("Parameters must satisfy 0 <= rho < mu")

        super().__init__(generator_matrix=ReedMullerCode._reed_muller_generator_matrix(rho, mu))
        self._minimum_distance = 2**(mu - rho)
        self._rho = rho
        self._mu = mu

    def __repr__(self):
        args = '{}, {}'.format(self._rho, self._mu)
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def rho(self):
        """
        The parameter :math:`\\rho` of the code. This property is read-only.
        """
        return self._rho

    @property
    def mu(self):
        """
        The parameter :math:`\\mu` of the code. This property is read-only.
        """
        return self._mu

    @functools.cached_property
    def reed_partitions(self):
        """
        The Reed partitions of the code. See :cite:`Lin.Costello.04` (p. 105--114) for details. This property is read-only.

        .. rubric:: Examples

        >>> code = komm.ReedMullerCode(2, 4)
        >>> code.reed_partitions[1]
        array([[ 0,  1,  4,  5],
               [ 2,  3,  6,  7],
               [ 8,  9, 12, 13],
               [10, 11, 14, 15]])
        >>> code.reed_partitions[8]
        array([[ 0,  4],
               [ 1,  5],
               [ 2,  6],
               [ 3,  7],
               [ 8, 12],
               [ 9, 13],
               [10, 14],
               [11, 15]])
        """
        reed_partitions = []
        for ell in range(self._rho, -1, -1):
            binary_vectors_I = np.fliplr(np.array(list(itertools.product([0, 1], repeat=ell)), dtype=int))
            binary_vectors_J = np.fliplr(np.array(list(itertools.product([0, 1], repeat=self._mu - ell)), dtype=int))
            for I in itertools.combinations(range(self._mu), ell):
                I = np.array(I, dtype=int)
                E = np.setdiff1d(np.arange(self._mu), I, assume_unique=True)
                S = np.dot(binary_vectors_I, 2**I)
                Q = np.dot(binary_vectors_J, 2**E)
                reed_partitions.append(S[np.newaxis] + Q[np.newaxis].T)
        return reed_partitions

    @staticmethod
    def _reed_muller_generator_matrix(rho, mu):
        """
        [1] Lin, Costello, 2Ed, p. 105--114. Assumes 0 <= rho < mu.
        """
        v = np.empty((mu, 2**mu), dtype=int)
        for i in range(mu):
            block = np.hstack((np.zeros(2**(mu - i - 1), dtype=int), np.ones(2**(mu - i - 1), dtype=int)))
            v[mu - i - 1] = np.tile(block, 2**i)

        G_list = []
        for ell in range(rho, 0, -1):
            for I in itertools.combinations(range(mu), ell):
                row = functools.reduce(np.multiply, v[I, :])
                G_list.append(row)
        G_list.append(np.ones(2**mu, dtype=int))

        return np.array(G_list, dtype=int)

    @tag(name='Reed', input_type='hard', target='message')
    def _decode_reed(self, recvword):
        """
        Reed decoding algorithm for Reed--Muller codes. It's a majority-logic decoding algorithm. See Lin, Costello, 2Ed, p. 105--114, 439--440.
        """
        message_hat = np.empty(self._generator_matrix.shape[0], dtype=int)
        bx = np.copy(recvword)
        for idx, partition in enumerate(self.reed_partitions):
            checksums = np.count_nonzero(bx[partition], axis=1) % 2
            message_hat[idx] = np.count_nonzero(checksums) > len(checksums) // 2
            bx ^= message_hat[idx] * self._generator_matrix[idx]
        return message_hat

    @tag(name='Weighted Reed', input_type='soft', target='message')
    def _decode_weighted_reed(self, recvword):
        """
        Weighted Reed decoding algorithm for Reed--Muller codes. See Lin, Costello, 2Ed, p. 440-442.
        """
        message_hat = np.empty(self._generator_matrix.shape[0], dtype=int)
        bx = (recvword < 0) * 1
        for idx, partition in enumerate(self.reed_partitions):
            checksums = np.count_nonzero(bx[partition], axis=1) % 2
            min_reliability = np.min(np.abs(recvword[partition]), axis=1)
            decision_var = np.dot(1 - 2*checksums, min_reliability)
            message_hat[idx] = decision_var < 0
            bx ^= message_hat[idx] * self._generator_matrix[idx]
        return message_hat

    def _default_decoder(self, dtype):
        if dtype == int:
            return 'reed'
        elif dtype == float:
            return 'weighted_reed'
