from functools import cache, cached_property

import numpy as np
import numpy.typing as npt
from attrs import field, frozen
from tqdm import tqdm

from komm._error_control_block import BlockCode
from komm._util.bit_operations import int2binlist


@frozen(eq=False)
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
    _codewords: list[int] = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if not 1 <= self.d <= self.n:
            raise ValueError("'n' and 'd' must satisfy 1 <= d <= n")
        object.__setattr__(self, "_codewords", self._generate_codewords())

    def _generate_codewords(self) -> list[int]:
        codewords = [0]
        for i in tqdm(range(1, 2**self.n), desc="Generating lexicode", delay=2.5):
            # Reverse checking is way faster [Wikipedia].
            if all((c ^ i).bit_count() >= self.d for c in reversed(codewords)):
                codewords.append(i)
        return codewords

    @property
    def length(self) -> int:
        return self.n

    @cache
    def minimum_distance(self) -> int:
        return self.d

    @cached_property
    def generator_matrix(self) -> npt.NDArray[np.int_]:
        n = self.n
        codewords = self._codewords
        k = len(codewords).bit_length() - 1
        generator_matrix = np.empty((k, n), dtype=int)
        for i in range(k):
            generator_matrix[i] = list(reversed(int2binlist(codewords[1 << i], n)))
        return generator_matrix
