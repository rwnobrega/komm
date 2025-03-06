from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt

from .util import Word, integer_to_symbols, symbols_to_integer


@dataclass
class LempelZivWelchCode:
    r"""
    Lempel–Ziv–Welch (LZW) code. It is a lossless data compression algorithm which is variation of the [Lempel–Ziv 78](/ref/LempelZiv78Code) algorithm. For more details, see <cite>Say06, Sec. 5.4.2</cite>.

    Parameters:
        source_cardinality: The source cardinality $S$. Must be an integer greater than or equal to $2$.
        target_cardinality: The target cardinality $T$. Must be an integer greater than or equal to $2$. Default is $2$ (binary).

    Examples:
        >>> lzw = komm.LempelZivWelchCode(2)  # Binary source, binary target
        >>> lzw = komm.LempelZivWelchCode(3, 4)  # Ternary source, quaternary target
    """

    source_cardinality: int
    target_cardinality: int = 2

    def __post_init__(self) -> None:
        if not self.source_cardinality >= 2:
            raise ValueError("'source_cardinality': must be at least 2")
        if not self.target_cardinality >= 2:
            raise ValueError("'target_cardinality': must be at least 2")

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a sequence of source symbols using the LZW encoding algorithm.

        Parameters:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code).

        Returns:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code).

        Examples:
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])

            >>> lzw = komm.LempelZivWelchCode(2, 8)
            >>> lzw.encode(np.zeros(15, dtype=int))
            array([0, 2, 3, 4, 5])
        """
        T = self.target_cardinality
        S = self.source_cardinality
        input = np.asarray(input)
        dictionary: dict[Word, int] = {(s,): s for s in range(S)}
        output: list[int] = []

        word: Word = ()
        for symbol in input:
            if word + (symbol,) in dictionary:
                word += (symbol,)
                continue
            k = ceil(log(len(dictionary), T))
            pointer = dictionary[word]
            output.extend(integer_to_symbols(pointer, base=T, width=k))
            dictionary[word + (symbol,)] = len(dictionary)
            word = (symbol,)

        if word:
            k = ceil(log(len(dictionary), T))
            pointer = dictionary[word]
            output.extend(integer_to_symbols(pointer, base=T, width=k))

        return np.array(output, dtype=int)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decodes a sequence of encoded symbols using the LZW decoding algorithm.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code). Also, the sequence must be a valid output of the `encode` method.

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code).

        Examples:
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.decode([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            >>> lzw = komm.LempelZivWelchCode(2, 8)
            >>> lzw.decode([0, 2, 3, 4, 5])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        T = self.target_cardinality
        S = self.source_cardinality
        input = np.asarray(input)

        if input.size == 0:
            return np.array([], dtype=int)

        dictionary: dict[int, Word] = {s: (s,) for s in range(S)}
        output: list[int] = []

        k = ceil(log(S, T))
        pointer = symbols_to_integer(input[:k], base=T)
        old = dictionary[pointer]
        output.extend(old)

        i = k
        while True:
            k = ceil(log(len(dictionary) + 1, T))
            if i + k > input.size:
                break
            pointer = symbols_to_integer(input[i : i + k], base=T)
            word = dictionary.get(pointer, old + (old[0],))
            output.extend(word)
            dictionary[len(dictionary)] = old + (word[0],)
            old = word
            i += k

        return np.array(output, dtype=int)
