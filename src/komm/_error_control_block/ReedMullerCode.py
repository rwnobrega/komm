from collections import Counter
from functools import cache, reduce
from itertools import combinations
from math import comb

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.bit_operations import int_to_bits
from .._util.docs import mkdocstrings
from .._util.matrices import row_span
from .._util.validators import validate_integer, validate_positive_integer
from .BlockCode import BlockCode
from .util import macwilliams_transform


@mkdocstrings(members=["reed_partitions"], filters=["!.*"])
class ReedMullerCode(BlockCode):
    r"""
    Reed–Muller code. Let $\mu$ and $\rho$ be two integers such that $0 \leq \rho < \mu$. The Reed–Muller code with parameters $(\rho, \mu)$ is the [linear block code](/ref/BlockCode) whose generator matrix rows are
    $$
        \mathbf{1}, \, v_1, \ldots, v_m, \, v_1 v_2, v_1 v_3, \ldots, v_{\mu} v_{\mu - 1}, \, \ldots \, \text{(up to products of degree $\rho$)},
    $$
    where $\mathbf{1}$ is the all-one $2^\mu$-tuple, $v_i$ is the binary $2^\mu$-tuple composed of $2^{\mu-i}$ repetitions of alternating blocks of zeros and ones, each block of length $2^i$, for $1 \leq i \leq \mu$, and $v_i v_j$ denotes the component-wise product (logical AND) of $v_i$ and $v_j$. Here we take the rows in reverse order. The resulting code has the following parameters:

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
        - The dual of the code with parameters $(\rho, \mu)$ is the code with parameters $(\mu - \rho - 1, \mu)$.

    Parameters:
        rho: The parameter $\rho$ of the code.
        mu: The parameter $\mu$ of the code.

    Examples:
        >>> code = komm.ReedMullerCode(2, 4)
        >>> (code.length, code.dimension, code.redundancy)
        (16, 11, 5)
        >>> code.generator_matrix
        array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
               [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
               [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        >>> code.minimum_distance()
        4
    """

    def __init__(self, rho: int, mu: int) -> None:
        mu = validate_positive_integer(mu)
        rho = validate_integer(rho, low=0, high=mu, rule="0 <= rho < mu")
        self.rho = rho
        self.mu = mu
        super().__init__(generator_matrix=reed_muller_generator_matrix(rho, mu))

    def __repr__(self) -> str:
        args = f"rho={self.rho}, mu={self.mu}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def codeword_weight_distribution(self) -> list[int]:
        rho, mu = self.rho, self.mu
        # Generic method: trivial for mu = 1, no gain for mu > 8.
        if not 2 <= mu <= 8:
            return super().codeword_weight_distribution()
        if self.redundancy < self.dimension:
            # The dual code is RM(mu - rho - 1, mu).
            dual = reed_muller_weight_distribution(mu - rho - 1, mu)
            return macwilliams_transform(dual)
        return reed_muller_weight_distribution(rho, mu)

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
        bin_vectors = [
            int_to_bits(range(2**ell), width=ell).reshape(2**ell, ell)
            for ell in range(mu + 1)
        ]
        for ell in range(rho, -1, -1):
            for indices in combinations(range(mu), ell):
                setI = np.array(indices, dtype=int)
                setE = np.setdiff1d(np.arange(mu), indices, assume_unique=True)
                setS = np.dot(bin_vectors[ell], 2**setI)
                setQ = np.dot(bin_vectors[mu - ell], 2**setE)
                reed_partitions.append(setS[np.newaxis] + setQ[np.newaxis].T)
        return reed_partitions


def reed_muller_generator_matrix(rho: int, mu: int) -> npt.NDArray[np.integer]:
    # See [LC04, pp. 105–114]. Assumes 0 <= rho < mu.
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


def reed_muller_weight_distribution(rho: int, mu: int) -> list[int]:
    # Codewords are (u | u + v), with u in RM(ρ, μ - 1) and v in RM(ρ - 1, μ - 1).
    # Both halves lie in the same coset C of RM(ρ - 1, μ - 1), and every pair of
    # words of C forms a codeword. Hence A(z) = sum_C φ_C(z)^2, where φ_C(z) is
    # the weight enumerator of C.
    rows = np.packbits(reed_muller_generator_matrix(rho, mu - 1), axis=1)
    top = comb(mu - 1, rho)  # Rows of degree rho come first.
    representatives, subcode = row_span(rows[:top]), row_span(rows[top:])
    enumerators: Counter[tuple[int, ...]] = Counter()
    desc = "Computing codeword weight distribution"
    for word in tqdm(representatives, desc=desc, delay=2.5):
        weights = np.bitwise_count(word ^ subcode).sum(-1, dtype=int)
        enumerator = np.bincount(weights, minlength=2 ** (mu - 1) + 1)
        enumerators[tuple(enumerator.tolist())] += 1
    # Few distinct enumerators: square each once.
    distribution = np.zeros(2**mu + 1, dtype=object)
    for enumerator, count in enumerators.items():
        poly = np.array(enumerator, dtype=object)
        distribution += count * np.convolve(poly, poly)
    return distribution.tolist()
