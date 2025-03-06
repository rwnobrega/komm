from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt

from .util import Word, integer_to_symbols, symbols_to_integer


@dataclass
class LempelZiv78Code:
    r"""
    Lempel–Ziv 78 (LZ78 or LZ2) code. It is a lossless data compression algorithm which is asymptotically optimal for ergodic sources. For more details, see <cite>Say06, Sec. 5.4.2</cite> and <cite>CT06, Sec. 13.4.2</cite>.

    Parameters:
        source_cardinality: The source cardinality $S$. Must be an integer greater than or equal to $2$.
        target_cardinality: The target cardinality $T$. Must be an integer greater than or equal to $2$. Default is $2$ (binary).

    Examples:
        >>> lz78 = komm.LempelZiv78Code(2)  # Binary source, binary target
        >>> lz78 = komm.LempelZiv78Code(3, 4)  # Ternary source, quaternary target
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
        Encodes a sequence of source symbols using the LZ78 encoding algorithm.

        Parameters:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code).

        Returns:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code).

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])

            >>> lz78 = komm.LempelZiv78Code(2, 8)
            >>> lz78.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 2, 0, 3, 0, 4, 0])
        """
        T = self.target_cardinality
        M = ceil(log(self.source_cardinality, T))
        input = np.asarray(input)
        dictionary: dict[Word, int] = {(): 0}
        output: list[int] = []

        word: Word = ()
        for symbol in input:
            if word + (symbol,) in dictionary:
                word += (symbol,)
                continue
            k = ceil(log(len(dictionary), T))
            pointer = dictionary[word]
            output.extend(integer_to_symbols(pointer, base=T, width=k))
            output.extend(integer_to_symbols(symbol, base=T, width=M))
            dictionary[word + (symbol,)] = len(dictionary)
            word = ()

        if word:
            k = ceil(log(len(dictionary), T))
            pointer = dictionary[word]
            output.extend(integer_to_symbols(pointer, base=T, width=k))

        return np.array(output, dtype=int)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decodes a sequence of encoded symbols using the LZ78 decoding algorithm.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code). Also, the sequence must be a valid output of the `encode` method.

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code).

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.decode([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            >>> lz78 = komm.LempelZiv78Code(2, 8)
            >>> lz78.decode([0, 1, 0, 2, 0, 3, 0, 4, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        T = self.target_cardinality
        M = ceil(log(self.source_cardinality, T))
        input = np.asarray(input)
        dictionary: dict[int, Word] = {0: ()}
        output: list[int] = []

        i = 0
        while True:
            k = ceil(log(len(dictionary), T))
            if i + k > input.size:
                break
            pointer = symbols_to_integer(input[i : i + k], base=T)
            word = dictionary[pointer]
            output.extend(word)
            i += k
            if i + M > input.size:
                break
            symbol = symbols_to_integer(input[i : i + M], base=T)
            output.append(symbol)
            dictionary[len(dictionary)] = word + (symbol,)
            i += M

        return np.array(output, dtype=int)
