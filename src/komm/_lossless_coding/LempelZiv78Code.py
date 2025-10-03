from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .util import Word, integer_to_symbols, symbols_to_integer


@dataclass
class LempelZiv78Code:
    r"""
    Lempel–Ziv 78 (LZ78 or LZ2) code. It is a lossless data compression algorithm which is asymptotically optimal for ergodic sources. For more details, see <cite>Say06, Sec. 5.4.2</cite> and <cite>CT06, Sec. 13.4.2</cite>.

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.

    Parameters:
        source_cardinality: The source cardinality $|\mathcal{X}|$. Must satisfy $|\mathcal{X}| \geq 2$.
        target_cardinality: The target cardinality $|\mathcal{Y}|$. Must satisfy $|\mathcal{Y}| \geq 2$. The default value is $2$ (binary).

    Examples:
        >>> lz78 = komm.LempelZiv78Code(2)  # Binary source, binary target
        >>> lz78 = komm.LempelZiv78Code(3, 4)  # Ternary source, quaternary target
    """

    source_cardinality: int
    target_cardinality: int = 2

    def __post_init__(self) -> None:
        if not self.source_cardinality >= 2:
            raise ValueError("'source_cardinality' must be at least 2")
        if not self.target_cardinality >= 2:
            raise ValueError("'target_cardinality' must be at least 2")

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a sequence of source symbols to a sequence of target symbols.

        Parameters:
            input: The sequence of source symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded target symbols. It is a 1D-array with elements in $\mathcal{Y}$.

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])

            >>> lz78 = komm.LempelZiv78Code(2, 8)
            >>> lz78.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 2, 0, 3, 0, 4, 0])
        """
        calY = self.target_cardinality
        M = ceil(log(self.source_cardinality, calY))
        input = np.asarray(input)
        dictionary: dict[Word, int] = {(): 0}
        output: list[int] = []

        word: Word = ()
        for symbol in tqdm(input, "Compressing LZ78", delay=2.5):
            if word + (symbol,) in dictionary:
                word += (symbol,)
                continue
            k = ceil(log(len(dictionary), calY))
            pointer = dictionary[word]
            output.extend(integer_to_symbols(pointer, base=calY, width=k))
            output.extend(integer_to_symbols(symbol, base=calY, width=M))
            dictionary[word + (symbol,)] = len(dictionary)
            word = ()

        if word:
            k = ceil(log(len(dictionary), calY))
            pointer = dictionary[word]
            output.extend(integer_to_symbols(pointer, base=calY, width=k))

        return np.array(output, dtype=int)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decodes a sequence of target symbols to a sequence of source symbols.

        Parameters:
            input: The sequence of target symbols to be decoded. Must be a 1D-array with elements in $\mathcal{Y}$. Also, the sequence must be a valid output of the `encode` method.

        Returns:
            output: The sequence of decoded source symbols. It is a 1D-array with elements in $\mathcal{X}$.

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.decode([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            >>> lz78 = komm.LempelZiv78Code(2, 8)
            >>> lz78.decode([0, 1, 0, 2, 0, 3, 0, 4, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        calY = self.target_cardinality
        M = ceil(log(self.source_cardinality, calY))
        input = np.asarray(input)
        dictionary: dict[int, Word] = {0: ()}
        output: list[int] = []

        pbar = tqdm(total=input.size, desc="Decompressing LZ78", delay=2.5)
        i = 0
        while True:
            k = ceil(log(len(dictionary), calY))
            if i + k > input.size:
                break
            pointer = symbols_to_integer(input[i : i + k], base=calY)
            word = dictionary[pointer]
            output.extend(word)
            i += k
            pbar.update(k)
            if i + M > input.size:
                break
            symbol = symbols_to_integer(input[i : i + M], base=calY)
            output.append(symbol)
            dictionary[len(dictionary)] = word + (symbol,)
            i += M
            pbar.update(M)
        pbar.close()

        return np.array(output, dtype=int)
