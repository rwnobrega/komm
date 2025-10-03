from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .util import Word, integer_to_symbols, symbols_to_integer


@dataclass
class LempelZivWelchCode:
    r"""
    Lempel–Ziv–Welch (LZW) code. It is a lossless data compression algorithm which is variation of the [Lempel–Ziv 78](/ref/LempelZiv78Code) algorithm. For more details, see <cite>Say06, Sec. 5.4.2</cite>.

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.

    Parameters:
        source_cardinality: The source cardinality $|\mathcal{X}|$. Must satisfy $|\mathcal{X}| \geq 2$.
        target_cardinality: The target cardinality $|\mathcal{Y}|$. Must satisfy $|\mathcal{Y}| \geq 2$. The default value is $2$ (binary).

    Examples:
        >>> lzw = komm.LempelZivWelchCode(2)  # Binary source, binary target
        >>> lzw = komm.LempelZivWelchCode(3, 4)  # Ternary source, quaternary target
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
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])

            >>> lzw = komm.LempelZivWelchCode(2, 8)
            >>> lzw.encode(np.zeros(15, dtype=int))
            array([0, 2, 3, 4, 5])
        """
        calX, calY = self.source_cardinality, self.target_cardinality
        input = np.asarray(input)
        dictionary: dict[Word, int] = {(s,): s for s in range(calX)}
        output: list[int] = []

        word: Word = ()
        for symbol in tqdm(input, "Compressing LZW", delay=2.5):
            if word + (symbol,) in dictionary:
                word += (symbol,)
                continue
            k = ceil(log(len(dictionary), calY))
            pointer = dictionary[word]
            output.extend(integer_to_symbols(pointer, base=calY, width=k))
            dictionary[word + (symbol,)] = len(dictionary)
            word = (symbol,)

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
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.decode([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            >>> lzw = komm.LempelZivWelchCode(2, 8)
            >>> lzw.decode([0, 2, 3, 4, 5])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        calX, calY = self.source_cardinality, self.target_cardinality
        input = np.asarray(input)

        if input.size == 0:
            return np.array([], dtype=int)

        dictionary: dict[int, Word] = {s: (s,) for s in range(calX)}
        output: list[int] = []

        k = ceil(log(calX, calY))
        pointer = symbols_to_integer(input[:k], base=calY)
        old = dictionary[pointer]
        output.extend(old)

        i = k
        pbar = tqdm(total=input.size, desc="Decompressing LZW", delay=2.5, initial=k)
        while True:
            k = ceil(log(len(dictionary) + 1, calY))
            if i + k > input.size:
                break
            pointer = symbols_to_integer(input[i : i + k], base=calY)
            word = dictionary.get(pointer, old + (old[0],))
            output.extend(word)
            dictionary[len(dictionary)] = old + (word[0],)
            old = word
            i += k
            pbar.update(k)
        pbar.close()

        return np.array(output, dtype=int)
