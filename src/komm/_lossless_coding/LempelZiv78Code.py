from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .. import abc
from .._util.validators import validate_integer, validate_integer_array
from .util import Word, integer_to_symbols, num_digits, symbols_to_integer

Token = tuple[int, int]


@dataclass
class LempelZiv78Code(abc.TokenCode[Token]):
    r"""
    Lempel–Ziv 78 (LZ78 or LZ2) code. It is a lossless data compression algorithm which is asymptotically optimal for ergodic sources. Let $\mathcal{X}$ be the source alphabet, and $\mathcal{Y}$ be the target alphabet. The token format is $(p, x)$, where $p \in \mathbb{N}$ is the index of the corresponding dictionary entry, and $x \in \mathcal{X}$ is the source symbol following the match. The index $p$ is represented as a variable-size word in $\mathcal{Y}^k$, where $k = \lceil \log_{|\mathcal{Y}|} i \rceil$, and $i$ is the size of the dictionary at the moment, including the empty word. For more details, see <cite>MacK03, Sec. 6.4</cite>, <cite>Say06, Sec. 5.4.2</cite> and <cite>CT06, Sec. 13.4.2</cite>.

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
        self.source_cardinality = validate_integer(
            self.source_cardinality, "source_cardinality", low=2
        )
        self.target_cardinality = validate_integer(
            self.target_cardinality, "target_cardinality", low=2
        )

    def source_to_tokens(self, source: npt.ArrayLike) -> list[Token]:
        r"""
        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.source_to_tokens([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])
            [(0, 1), (0, 0), (1, 1), (2, 1), (4, 0), (2, 0)]
        """
        calX = self.source_cardinality
        source = validate_integer_array(source, low=0, high=calX)
        dictionary: dict[Word, int] = {(): 0}
        tokens: list[Token] = []
        word: Word = ()
        for symbol in tqdm(source, "Compressing LZ78", delay=2.5):
            if word + (symbol,) in dictionary:
                word += (symbol,)
                continue
            p, x = dictionary[word], int(symbol)
            tokens.append((p, x))
            dictionary[word + (symbol,)] = len(dictionary)
            word = ()
        if word:
            tokens.append((dictionary[word], -1))
        return tokens

    def tokens_to_source(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.tokens_to_source([(0, 1), (0, 0), (1, 1), (2, 1), (4, 0), (2, 0)])
            array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])
        """
        dictionary: dict[int, Word] = {0: ()}
        source: list[int] = []
        for p, x in tokens:
            word = dictionary[p]
            source.extend(word)
            if x >= 0:
                source.append(x)
            dictionary[len(dictionary)] = word + (x,)
        return np.array(source, dtype=int)

    def tokens_to_target(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.tokens_to_target([(0, 1), (0, 0), (1, 1), (2, 1), (4, 0), (2, 0)])
            array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
        """
        calX, calY = self.source_cardinality, self.target_cardinality
        M = num_digits(calX, calY)
        target: list[int] = []
        for i, (p, x) in enumerate(tokens):
            k = num_digits(i + 1, calY)
            target.extend(integer_to_symbols(p, base=calY, width=k))
            if x >= 0:
                target.extend(integer_to_symbols(x, base=calY, width=M))
        return np.array(target, dtype=int)

    def target_to_tokens(self, target: npt.ArrayLike) -> list[Token]:
        r"""
        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.target_to_tokens(
            ...     [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
            ... )
            [(0, 1), (0, 0), (1, 1), (2, 1), (4, 0), (2, 0)]
        """
        calX, calY = self.source_cardinality, self.target_cardinality
        M = num_digits(calX, calY)
        target = np.asarray(target, dtype=int)
        tokens: list[Token] = []
        i = 0
        while i < target.size:
            k = num_digits(len(tokens) + 1, calY)
            p = int(symbols_to_integer(target[i : i + k], base=calY))
            i += k
            if i < target.size:
                x = int(symbols_to_integer(target[i : i + M], base=calY))
            else:
                x = -1
            tokens.append((p, x))
            i += M
        return tokens

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.encode([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])
            array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])

            >>> lz78 = komm.LempelZiv78Code(2, 8)
            >>> lz78.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 2, 0, 3, 0, 4, 0])
        """
        return super().encode(input)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.decode([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
            array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])

            >>> lz78 = komm.LempelZiv78Code(2, 8)
            >>> lz78.decode([0, 1, 0, 2, 0, 3, 0, 4, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        return super().decode(input)
