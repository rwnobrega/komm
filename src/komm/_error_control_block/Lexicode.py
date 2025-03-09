from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.bit_operations import int_to_bits
from .._util.docs import mkdocstrings
from .BlockCode import BlockCode


@mkdocstrings(filters=["!.*"])
@dataclass(eq=False)
class Lexicode(BlockCode):
    r"""
    Lexicographic code (lexicode). For a given length $n$ and minimum distance $d$, it is the [linear block code](/ref/BlockCode) obtained by starting with the all-zero codeword and adding all binary $n$-tuples (in lexicographic order) that are at least at distance $d$ from all codewords already in the code.

    Attributes:
        n: The length $n$ of the code.
        d: The minimum distance $d$ of the code.

    For more details, see <cite>HP03, Sec. 2.11</cite>.

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

    n: int
    d: int

    def __post_init__(self) -> None:
        if not 1 <= self.d <= self.n:
            raise ValueError("'n' and 'd' must satisfy 1 <= d <= n")
        super().__init__(generator_matrix=lexicode_generator_matrix(self.n, self.d))

    @cache
    def minimum_distance(self) -> int:
        return self.d


def lexicode_generator_matrix(n: int, d: int) -> npt.NDArray[np.integer]:
    codewords = [0]
    basis: list[int] = []
    for i in tqdm(range(1, 2**n), desc="Generating lexicode", delay=2.5):
        # Reverse checking is way faster [Wikipedia].
        if all((c ^ i).bit_count() >= d for c in reversed(codewords)):
            if len(codewords).bit_count() == 1:  # Is power of 2.
                basis.append(i)
            codewords.append(i)
    generator_matrix = np.array([list(reversed(int_to_bits(c, n))) for c in basis])
    return generator_matrix
