from math import ceil, log

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.validators import validate_integer_range
from .util import integer_to_symbols, symbols_to_integer

Token = tuple[int, int, int]


class LempelZiv77Code:
    r"""
    Lempel–Ziv 77 (LZ77 or LZ1) code. It is a lossless data compression algorithm which is asymptotically optimal for ergodic sources. Let $\mathcal{X}$ be the source alphabet, and $\mathcal{Y}$ be the target alphabet. The notation used here is the following: $S \geq 1$ is the size of the *search buffer*, $L \geq 1$ is the size of the *lookahead buffer*, and $W = S + L$ is the size of the *sliding window*. The token format follows the [original LZ77 paper](https://doi.org/10.1109%2FTIT.1977.1055714), namely $(p, \ell, x)$, where $p \in [0 : S)$ is the *pointer* for the match, $\ell \in [0 : L)$ is the *length* of the match, and $x \in \mathcal{X}$ is the source symbol following the match, but with both $p$ and $\ell$ being $0$-indexed instead of $1$-indexed. Also following the LZ77 original paper, a token is represented as a fixed-size word in $\mathcal{Y}^n$, where $$n = \log S + \log L + \log |\mathcal{X}|$$ and all logs are to base $|\mathcal{Y}|$. For more details, see <cite>Say06, Sec. 5.4.1</cite> and <cite>CT06, Sec. 13.4.1</cite>.

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.

    Parameters:
        search_size: The search buffer size $S$. Must satisfy $S \geq 1$.
        lookahead_size: The lookahead buffer size $L$. Must satisfy $L \geq 1$.
        source_cardinality: The source cardinality $|\mathcal{X}|$. Must satisfy $|\mathcal{X}| \geq 2$.
        target_cardinality: The target cardinality $|\mathcal{Y}|$. Must satisfy $|\mathcal{Y}| \geq 2$. The default value is $2$ (binary).
        search_buffer: The initial state of the search buffer. Must be a 1D-array of length $S$ with elements in $\mathcal{X}$. The default value corresponds to $(0, \ldots, 0) \in \mathcal{X}^S$.

    Examples:
        >>> lz77 = komm.LempelZiv77Code(
        ...     search_size=2**13,
        ...     lookahead_size=16,
        ...     source_cardinality=256,
        ...     target_cardinality=2,
        ... )
    """

    def __init__(
        self,
        *,
        search_size: int,
        lookahead_size: int,
        source_cardinality: int,
        target_cardinality: int = 2,
        search_buffer: npt.ArrayLike | None = None,
    ):
        if not search_size >= 1:
            raise ValueError("'search_size' must be at least 1")
        if not lookahead_size >= 1:
            raise ValueError("'lookahead_size' must be at least 1")
        if not source_cardinality >= 2:
            raise ValueError("'source_cardinality' must be at least 2")
        if not target_cardinality >= 2:
            raise ValueError("'target_cardinality' must be at least 2")

        self.search_size = search_size
        self.lookahead_size = lookahead_size
        self.source_cardinality = source_cardinality
        self.target_cardinality = target_cardinality

        if search_buffer is None:
            self.search_buffer = [0] * self.search_size
        else:
            self.search_buffer = np.asarray(search_buffer, dtype=int).tolist()
            if len(self.search_buffer) != self.search_size:
                raise ValueError(
                    "length of 'search_buffer' must be S ="
                    f" {self.search_size} (got {len(self.search_buffer)})"
                )

    def __repr__(self) -> str:
        args = ", ".join([
            f"search_size={self.search_size}",
            f"lookahead_size={self.lookahead_size}",
            f"source_cardinality={self.source_cardinality}",
            f"target_cardinality={self.target_cardinality}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    def window_size(self) -> int:
        r"""
        The sliding window size $W$. It is given by $W = S + L$.
        """
        return self.search_size + self.lookahead_size

    def _get_widths(self) -> tuple[int, int, int]:
        calY = self.target_cardinality
        p_width = ceil(log(self.search_size, calY))
        l_width = ceil(log(self.lookahead_size, calY))
        x_width = ceil(log(self.source_cardinality, calY))
        return p_width, l_width, x_width

    def source_to_tokens(self, source: npt.ArrayLike) -> list[Token]:
        r"""
        Encodes a given sequence of source symbols to the corresponding list of tokens.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.source_to_tokens([0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2])
            [(8, 2, 1), (7, 3, 2), (6, 7, 2)]
        """
        source = validate_integer_range(source, high=self.source_cardinality)
        ss, ls = self.search_size, self.lookahead_size
        buffer = bytes(self.search_buffer + source.tolist())
        tokens: list[Token] = []

        pbar = tqdm(total=len(buffer), initial=ss, desc="Compressing LZ77", delay=2.5)
        i = ss
        while i < len(buffer):
            l = min(ls, len(buffer) - i) - 1
            while True:
                p = buffer[i - ss : i + l - 1].rfind(buffer[i : i + l])
                if p >= 0:
                    break
                l -= 1
            tokens.append((p, l, buffer[i + l]))
            i += l + 1
            pbar.update(l + 1)

        return tokens

    def tokens_to_source(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Decodes a given list of tokens to the corresponding sequence of source symbols.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.tokens_to_source([(8, 2, 1), (7, 3, 2), (6, 7, 2)])
            array([0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2])
        """
        ss = self.search_size
        buffer = self.search_buffer.copy()
        for p, l, x in tokens:
            start = len(buffer) - ss + p
            for j in range(l):
                buffer.append(buffer[start + j])
            buffer.append(x)
        source = np.array(buffer[ss:], dtype=int)
        return source

    def tokens_to_target(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Returns the target alphabet representation corresponding to a given list of tokens.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.tokens_to_target([(8, 2, 1), (7, 3, 2), (6, 7, 2)])
            array([2, 2, 0, 2, 1, 2, 1, 1, 0, 2, 2, 0, 2, 1, 2])
        """
        calY = self.target_cardinality
        p_width, l_width, x_width = self._get_widths()
        target: list[int] = []
        for p, l, x in tokens:
            target.extend(integer_to_symbols(p, base=calY, width=p_width))
            target.extend(integer_to_symbols(l, base=calY, width=l_width))
            target.extend(integer_to_symbols(x, base=calY, width=x_width))
        return np.array(target, dtype=int)

    def target_to_tokens(self, target: npt.ArrayLike) -> list[Token]:
        r"""
        Returns the list of tokens corresponding to a given target alphabet representation.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.target_to_tokens([2, 2, 0, 2, 1, 2, 1, 1, 0, 2, 2, 0, 2, 1, 2])
            [(8, 2, 1), (7, 3, 2), (6, 7, 2)]
        """
        target = np.asarray(target, dtype=int)
        calY = self.target_cardinality
        p_width, l_width, x_width = self._get_widths()
        tokens: list[Token] = []
        i = 0
        while i + p_width + l_width + x_width <= target.size:
            p = int(symbols_to_integer(target[i : i + p_width], base=calY))
            i += p_width
            l = int(symbols_to_integer(target[i : i + l_width], base=calY))
            i += l_width
            x = int(symbols_to_integer(target[i : i + x_width], base=calY))
            i += x_width
            tokens.append((p, l, x))
        return tokens

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a sequence of source symbols to a sequence of target symbols.

        Parameters:
            input: The sequence of source symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded target symbols. It is a 1D-array with elements in $\mathcal{Y}$.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.encode([0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2])
            array([2, 2, 0, 2, 1, 2, 1, 1, 0, 2, 2, 0, 2, 1, 2])
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
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.decode([2, 2, 0, 2, 1, 2, 1, 1, 0, 2, 2, 0, 2, 1, 2])
            array([0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2])
        """
        tokens = self.target_to_tokens(input)
        output = self.tokens_to_source(tokens)
        return output
