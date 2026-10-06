from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .. import abc
from .._util.validators import validate_integer_array
from .util import Word, integer_to_symbols, num_digits, symbols_to_integer

Token = int


@dataclass
class LempelZivWelchCode(abc.TokenCode[Token]):
    r"""
    Lempel–Ziv–Welch (LZW) code. It is a lossless data compression algorithm which is a variation of the [Lempel–Ziv 78](/ref/LempelZiv78Code) algorithm. Let $\mathcal{X}$ be the source alphabet, and $\mathcal{Y}$ be the target alphabet. The token format is $p \in \mathbb{N}$, the index of the corresponding dictionary entry. The index $p$ is represented as a variable-size word in $\mathcal{Y}^k$, where $k = \lceil \log_{|\mathcal{Y}|} i \rceil$, and $i$ is the size of the dictionary at the moment. For more details, see <cite>Say06, Sec. 5.4.2</cite>.

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

    def _width(self, i: int) -> int:
        # Dictionary has |X| + i entries at the i-th token
        return num_digits(self.source_cardinality + i, self.target_cardinality)

    def source_to_tokens(self, source: npt.ArrayLike) -> list[Token]:
        r"""
        Examples:
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.source_to_tokens(np.zeros(15, dtype=int))
            [0, 2, 3, 4, 5]
        """
        calX = self.source_cardinality
        source = validate_integer_array(source, "source", low=0, high=calX)
        dictionary: dict[Word, int] = {(s,): s for s in range(self.source_cardinality)}
        tokens: list[Token] = []
        word: Word = ()
        for symbol in tqdm(source, "Compressing LZW", delay=2.5):
            if word + (symbol,) in dictionary:
                word += (symbol,)
                continue
            tokens.append(dictionary[word])
            dictionary[word + (symbol,)] = len(dictionary)
            word = (symbol,)
        if word:
            tokens.append(dictionary[word])
        return tokens

    def tokens_to_source(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.tokens_to_source([0, 2, 3, 4, 5])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        dictionary: dict[int, Word] = {s: (s,) for s in range(self.source_cardinality)}
        source: list[int] = []
        old: Word = ()
        for pointer in tqdm(tokens, "Decompressing LZW", delay=2.5):
            word = dictionary.get(pointer, old + old[:1])
            source.extend(word)
            if old:
                dictionary[len(dictionary)] = old + word[:1]
            old = word
        return np.array(source, dtype=int)

    def tokens_to_target(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.tokens_to_target([0, 2, 3, 4, 5])
            array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
        """
        calY = self.target_cardinality
        target: list[int] = []
        for i, pointer in enumerate(tokens):
            target.extend(integer_to_symbols(pointer, base=calY, width=self._width(i)))
        return np.array(target, dtype=int)

    def target_to_tokens(self, target: npt.ArrayLike) -> list[Token]:
        r"""
        Examples:
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.target_to_tokens([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
            [0, 2, 3, 4, 5]
        """
        target = np.asarray(target, dtype=int)
        calY = self.target_cardinality
        tokens: list[Token] = []
        i = 0
        while i + self._width(len(tokens)) <= target.size:
            k = self._width(len(tokens))
            tokens.append(symbols_to_integer(target[i : i + k], base=calY))
            i += k
        return tokens

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])

            >>> lzw = komm.LempelZivWelchCode(2, 8)
            >>> lzw.encode(np.zeros(15, dtype=int))
            array([0, 2, 3, 4, 5])
        """
        return super().encode(input)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> lzw = komm.LempelZivWelchCode(2)
            >>> lzw.decode([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            >>> lzw = komm.LempelZivWelchCode(2, 8)
            >>> lzw.decode([0, 2, 3, 4, 5])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        return super().decode(input)
