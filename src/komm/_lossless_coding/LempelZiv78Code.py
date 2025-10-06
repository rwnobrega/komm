from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .util import Word, integer_to_symbols, symbols_to_integer

Token = tuple[int, int]


@dataclass
class LempelZiv78Code:
    r"""
    Lempel–Ziv 78 (LZ78 or LZ2) code. It is a lossless data compression algorithm which is asymptotically optimal for ergodic sources. Let $\mathcal{X}$ be the source alphabet, and $\mathcal{Y}$ be the target alphabet. The token format is $(p, x)$, where $p \in \mathbb{N}$ is the index of the corresponding dictionary entry, and $x \in \mathcal{X}$ is the source symbol following the match. The index $p$ is represented as a variable-size word in $\mathcal{Y}^k$, where $k = \log_{|\mathcal{Y}|}(i + 1)$, and $i$ is the size of the dictionary at the moment. For more details, see <cite>MacK03, Sec. 6.4</cite>, <cite>Say06, Sec. 5.4.2</cite> and <cite>CT06, Sec. 13.4.2</cite>.

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

    def source_to_tokens(self, source: npt.ArrayLike) -> list[Token]:
        r"""
        Encodes a given sequence of source symbols to the corresponding list of tokens.

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.source_to_tokens([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])
            [(0, 1), (0, 0), (1, 1), (2, 1), (4, 0), (2, 0)]
        """
        source = np.asarray(source)
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
        Decodes a given list of tokens to the corresponding sequence of source symbols.

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
        Returns the target alphabet representation corresponding to a given list of tokens.

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.tokens_to_target([(0, 1), (0, 0), (1, 1), (2, 1), (4, 0), (2, 0)])
            array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
        """
        calX, calY = self.source_cardinality, self.target_cardinality
        M = ceil(log(calX, calY))
        target: list[int] = []
        for i, (p, x) in enumerate(tokens):
            k = ceil(log(i + 1, calY))
            target.extend(integer_to_symbols(p, base=calY, width=k))
            if x >= 0:
                target.extend(integer_to_symbols(x, base=calY, width=M))
        return np.array(target, dtype=int)

    def target_to_tokens(self, target: npt.ArrayLike) -> list[Token]:
        r"""
        Returns the list of tokens corresponding to a given target alphabet representation.

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.target_to_tokens([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
            [(0, 1), (0, 0), (1, 1), (2, 1), (4, 0), (2, 0)]
        """
        calX, calY = self.source_cardinality, self.target_cardinality
        M = ceil(log(calX, calY))
        target = np.asarray(target)
        tokens: list[Token] = []
        i = 0
        while i < target.size:
            k = ceil(log(len(tokens) + 1, calY))
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
        Encodes a sequence of source symbols to a sequence of target symbols.

        Parameters:
            input: The sequence of source symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded target symbols. It is a 1D-array with elements in $\mathcal{Y}$.

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.encode([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])
            array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])

            >>> lz78 = komm.LempelZiv78Code(2, 8)
            >>> lz78.encode(np.zeros(15, dtype=int))
            array([0, 1, 0, 2, 0, 3, 0, 4, 0])
        """
        tokens = self.source_to_tokens(input)
        output = self.tokens_to_target(tokens)
        return output

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decodes a sequence of target symbols to a sequence of source symbols.

        Parameters:
            input: The sequence of target symbols to be decoded. Must be a 1D-array with elements in $\mathcal{Y}$. Also, the sequence must be a valid output of the `encode` method.

        Returns:
            output: The sequence of decoded source symbols. It is a 1D-array with elements in $\mathcal{X}$.

        Examples:
            >>> lz78 = komm.LempelZiv78Code(2)
            >>> lz78.decode([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
            array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])

            >>> lz78 = komm.LempelZiv78Code(2, 8)
            >>> lz78.decode([0, 1, 0, 2, 0, 3, 0, 4, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """
        tokens = self.target_to_tokens(input)
        output = self.tokens_to_source(tokens)
        return output
