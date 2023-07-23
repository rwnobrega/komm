import itertools as it
from functools import cached_property, reduce

import numpy as np
from attrs import frozen

from .BlockCode import BlockCode


@frozen(slots=False)
class ReedMullerCode(BlockCode):
    r"""
    Reed–Muller code. It is a [linear block code](/ref/BlockCode) defined by two integers $\rho$ and $\mu$, which must satisfy $0 \leq \rho < \mu$. See references for more details. The resulting code is denoted by $\mathrm{RM}(\rho, \mu)$, and has the following parameters:

    - Length: $n = 2^{\mu}$
    - Dimension: $k = 1 + {\mu \choose 1} + \cdots + {\mu \choose \rho}$
    - Redundancy: $m = 1 + {\mu \choose 1} + \cdots + {\mu \choose \mu - \rho - 1}$
    - Minimum distance: $d = 2^{\mu - \rho}$

    For more details, see <cite>LC04, Sec. 4.3</cite>.

    Notes:

        - For $\rho = 0$ it reduces to a [repetition code](/ref/RepetitionCode).
        - For $\rho = 1$ it reduces to a lengthened [simplex code](/ref/SimplexCode).
        - For $\rho = \mu - 2$ it reduces to an extended [Hamming code](/ref/HammingCode).
        - For $\rho = \mu - 1$ it reduces to a [single parity check code](/ref/SingleParityCheckCode).

    Attributes:

        rho: The parameter $\rho$ of the code.
        mu: The parameter $\mu$ of the code.

    The parameters must satisfy $0 \leq \rho < \mu$.

    Examples:

        >>> code = komm.ReedMullerCode(1, 5)
        >>> (code.length, code.dimension, code.redundancy)
        (32, 6, 26)
        >>> code.minimum_distance
        16
        >>> code.generator_matrix  #doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
               [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    """
    rho: int
    mu: int

    def __post_init__(self):
        if not 0 <= self.rho < self.mu:
            raise ValueError("'rho' and 'mu' must satisfy 0 <= rho < mu")

    @cached_property
    def generator_matrix(self):
        # See [LC04, p. 105–114]. Assumes 0 <= rho < mu.
        rho, mu = self.rho, self.mu
        v = np.empty((mu, 2**mu), dtype=int)
        for i in range(mu):
            block = np.hstack((np.zeros(2 ** (mu - i - 1), dtype=int), np.ones(2 ** (mu - i - 1), dtype=int)))
            v[mu - i - 1] = np.tile(block, 2**i)

        G_list = []
        for ell in range(rho, 0, -1):
            for I in it.combinations(range(mu), ell):
                row = reduce(np.multiply, v[I, :])
                G_list.append(row)
        G_list.append(np.ones(2**mu, dtype=int))

        return np.array(G_list, dtype=int)

    @property
    def minimum_distance(self):
        return 2 ** (self.mu - self.rho)

    @property
    def default_decoder(self):
        return "reed"

    @classmethod
    @property
    def supported_decoders(cls):
        return cls.__base__.supported_decoders + ["reed", "weighted_reed"]  # type: ignore

    @property
    def reed_partitions(self):
        r"""
        The Reed partitions of the code. See <cite>LC04, Sec. 4.3</cite>.

        Examples:

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
        rho, mu = self.rho, self.mu
        reed_partitions = []
        for ell in range(rho, -1, -1):
            binary_vectors_I = np.fliplr(np.array(list(it.product([0, 1], repeat=ell)), dtype=int))
            binary_vectors_J = np.fliplr(np.array(list(it.product([0, 1], repeat=mu - ell)), dtype=int))
            for I in it.combinations(range(mu), ell):
                I = np.array(I, dtype=int)
                E = np.setdiff1d(np.arange(mu), I, assume_unique=True)
                S = np.dot(binary_vectors_I, 2**I)
                Q = np.dot(binary_vectors_J, 2**E)
                reed_partitions.append(S[np.newaxis] + Q[np.newaxis].T)
        return reed_partitions
