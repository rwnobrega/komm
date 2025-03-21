from dataclasses import dataclass
from functools import cache, reduce
from itertools import combinations, product

import numpy as np
import numpy.typing as npt

from .._util.docs import mkdocstrings
from .BlockCode import BlockCode


@mkdocstrings(members=["reed_partitions"], filters=["!.*"])
@dataclass(eq=False)
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
        - For $\rho = \mu - 1$ it reduces to a [single parity-check code](/ref/SingleParityCheckCode).

    Attributes:
        rho: The parameter $\rho$ of the code.
        mu: The parameter $\mu$ of the code.

    The parameters must satisfy $0 \leq \rho < \mu$.

    Examples:
        >>> code = komm.ReedMullerCode(1, 5)
        >>> (code.length, code.dimension, code.redundancy)
        (32, 6, 26)
        >>> code.generator_matrix
        array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
               [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        >>> code.minimum_distance()
        16
    """

    rho: int
    mu: int

    def __post_init__(self) -> None:
        if not 0 <= self.rho < self.mu:
            raise ValueError("'rho' and 'mu' must satisfy 0 <= rho < mu")
        super().__init__(
            generator_matrix=reed_muller_generator_matrix(self.rho, self.mu)
        )

    @cache
    def minimum_distance(self) -> int:
        return 2 ** (self.mu - self.rho)

    @cache
    def reed_partitions(self) -> list[npt.NDArray[np.integer]]:
        r"""
        The Reed partitions of the code. See <cite>LC04, Sec. 4.3</cite>.

        Examples:
            >>> code = komm.ReedMullerCode(2, 4)
            >>> reed_partitions = code.reed_partitions()
            >>> reed_partitions[1]
            array([[ 0,  1,  4,  5],
                   [ 2,  3,  6,  7],
                   [ 8,  9, 12, 13],
                   [10, 11, 14, 15]])
            >>> reed_partitions[8]
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
        reed_partitions: list[npt.NDArray[np.integer]] = []
        binary_vectors = [
            np.fliplr(np.array(list(product([0, 1], repeat=ell)), dtype=int))
            for ell in range(mu + 1)
        ]
        for ell in range(rho, -1, -1):
            for indices in combinations(range(mu), ell):
                setI = np.array(indices, dtype=int)
                setE = np.setdiff1d(np.arange(mu), indices, assume_unique=True)
                setS = np.dot(binary_vectors[ell], 2**setI)
                setQ = np.dot(binary_vectors[mu - ell], 2**setE)
                reed_partitions.append(setS[np.newaxis] + setQ[np.newaxis].T)
        return reed_partitions


def reed_muller_generator_matrix(rho: int, mu: int) -> npt.NDArray[np.integer]:
    # See [LC04, p. 105–114]. Assumes 0 <= rho < mu.
    v = np.empty((mu, 2**mu), dtype=int)
    for i in range(mu):
        block = np.hstack((
            np.zeros(2 ** (mu - i - 1), dtype=int),
            np.ones(2 ** (mu - i - 1), dtype=int),
        ))
        v[mu - i - 1] = np.tile(block, 2**i)

    G_list: list[npt.NDArray[np.integer]] = []
    for ell in range(rho, 0, -1):
        for indices in combinations(range(mu), ell):
            row = reduce(np.multiply, v[indices, :])
            G_list.append(row)
    row = np.ones(2**mu, dtype=int)
    G_list.append(row)

    return np.array(G_list, dtype=int)
