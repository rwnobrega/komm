from functools import cache

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.bit_operations import int_to_bits
from .._util.docs import mkdocstrings
from .._util.validators import validate_integer
from .BlockCode import BlockCode


@mkdocstrings(filters=["!.*"])
class Lexicode(BlockCode):
    r"""
    Lexicographic code (lexicode). For a given length $n$ and minimum distance $d$, it is the [linear block code](/ref/BlockCode) obtained by starting with the all-zero codeword and adding all binary $n$-tuples (in lexicographic order) that are at least at distance $d$ from all codewords already in the code. For more details, see <cite>HP03, Sec. 2.11</cite>.

    Notes:
        - For $d = 2$ it reduces to the [single parity-check code](/ref/SingleParityCheckCode) of length $n$.
        - For $d = n$ it reduces to the [repetition code](/ref/RepetitionCode) of length $n$.
        - For $n = 2^\mu - 1$ and $d = 3$ it is equivalent to the [Hamming code](/ref/HammingCode); for $n = 2^\mu$ and $d = 4$, to its extended version.
        - For $n = 23$ and $d = 7$ it is equivalent to the [Golay code](/ref/GolayCode); for $n = 24$ and $d = 8$, to its extended version.

    Parameters:
        n: The length $n$ of the code.
        d: The minimum distance $d$ of the code. Must satisfy $1 \leq d \leq n$.

    **Tables:**

    - [Table of dimensions.](/tables/lexicodes/#dimensions)

    Examples:
        >>> code = komm.Lexicode(7, 3)  # Hamming (7, 4)
        >>> (code.length, code.dimension, code.redundancy)
        (7, 4, 3)
        >>> code.generator_matrix
        array([[0, 0, 0, 0, 1, 1, 1],
               [0, 0, 1, 1, 0, 0, 1],
               [0, 1, 0, 1, 0, 1, 0],
               [1, 0, 0, 1, 0, 1, 1]])
        >>> code.minimum_distance()
        3
    """

    def __init__(self, n: int, d: int) -> None:
        n = validate_integer(n, "n", low=1)
        d = validate_integer(d, "d", low=1, high=n + 1)
        self.n = n
        self.d = d
        super().__init__(generator_matrix=lexicode_generator_matrix(n, d))

    def __repr__(self) -> str:
        args = f"n={self.n}, d={self.d}"
        return f"{self.__class__.__name__}({args})"

    @cache
    def minimum_distance(self) -> int:
        return self.d


def lexicode_generator_matrix(n: int, d: int) -> npt.NDArray[np.integer]:
    # Per syndrome: coset weight and smallest word.
    weight = np.array([0], dtype=np.uint8)
    leader = np.array([0])
    basis: list[int] = []
    for m in tqdm(range(n), desc="Generating lexicode", delay=2.5):
        # Cosets at distance at least d - 1.
        far = weight >= d - 1
        if not far.any():  # New check bit.
            weight = np.concatenate([weight, weight + 1])
            leader = np.concatenate([leader, leader + 2**m])
        else:
            h = np.argmax(far)  # First far syndrome.
            basis.append(2**m + int(leader[h]))
            weight = np.minimum(weight, weight[np.arange(weight.size) ^ h] + 1)
    generator_matrix = int_to_bits(basis, width=n, bit_order="MSB-first").reshape(-1, n)
    return generator_matrix
