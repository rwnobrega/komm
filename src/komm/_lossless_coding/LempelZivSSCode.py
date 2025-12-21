from math import ceil, log
from typing import Literal

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.validators import validate_integer_range
from .util import integer_to_symbols, symbols_to_integer

Token = tuple[Literal[0], int] | tuple[Literal[1], int, int]


class LempelZivSSCode:
    r"""
    Lempel–Ziv–Storer–Szymanski (LZSS) code. It is a lossless data compression algorithm which is a variation of the [Lempel–Ziv 77](/ref/LempelZiv77Code) algorithm. Let $\mathcal{X}$ be the source alphabet, $\mathcal{Y}$ be the target alphabet, $S \geq 1$ be the size of the *search buffer*, and $L \geq 1$ be the size of the *lookahead buffer*. The token format follows <cite>CT06, Sec. 13.4.1</cite>, where a token is either a *literal* $(0, x)$, where $x \in \mathcal{X}$ is a source symbol, or a *reference* $(1, p, \ell)$, where $p \in [1 : S]$ is the *pointer* (location of the beginning of the match, measured backward from the end of the search window), and $\ell \in [1 : L]$ is the *length* of the match. References are only encoded if they provide compression benefit (length exceeds the break-even point). For more details, see <cite>CT06, Sec. 13.4.1</cite>.

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.

    Parameters:
        search_size: The search buffer size $S$. Must satisfy $S \geq 1$.
        lookahead_size: The lookahead buffer size $L$. Must satisfy $L \geq 1$.
        source_cardinality: The source cardinality $|\mathcal{X}|$. Must satisfy $|\mathcal{X}| \geq 2$.
        target_cardinality: The target cardinality $|\mathcal{Y}|$. Must satisfy $|\mathcal{Y}| \geq 2$. The default value is $2$ (binary).
        search_buffer: The initial state of the search buffer. Must be a 1D-array of length $S$ with elements in $\mathcal{X}$. The default value corresponds to $(0, \ldots, 0) \in \mathcal{X}^S$.

    Examples:
        >>> lzss = komm.LempelZivSSCode(
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

    @property
    def break_even(self) -> int:
        r"""
        The break-even point for encoding a reference. A match must be at least this long to be worth encoding as a reference instead of literals. It is given by
        $$
            \left\lceil \frac{1 + \lceil \log S \rceil + \lceil \log L \rceil}{1+ \lceil \log |\mathcal{X}| \rceil} \right\rceil,
        $$
        where all logs are to base $|\mathcal{Y}|$.
        """
        p_width, l_width, x_width = self._get_widths()
        # Cost as a reference: 1 (flag) + p_width + l_width
        # Cost as literals: length * (1 (flag) + x_width)
        # Break even when: 1 + p_width + l_width < length * (1 + x_width)
        return ceil((1 + p_width + l_width) / (1 + x_width))

    def source_to_tokens(self, source: npt.ArrayLike) -> list[Token]:
        r"""
        Encodes a given sequence of source symbols to the corresponding list of tokens.

        Examples:
            >>> lzss = komm.LempelZivSSCode(
            ...     search_size=8,
            ...     lookahead_size=4,
            ...     source_cardinality=4,
            ... )
            >>> lzss.source_to_tokens([3, 0, 0, 0, 3, 2, 0, 3])
            [(0, 3), (1, 4, 4), (0, 2), (1, 3, 2)]
        """
        source = validate_integer_range(source, high=self.source_cardinality)
        ss, ls = self.search_size, self.lookahead_size
        break_even = self.break_even
        buffer = bytes(self.search_buffer + source.tolist())
        tokens: list[Token] = []

        pbar = tqdm(total=len(buffer), initial=ss, desc="Compressing LZSS", delay=2.5)
        i = ss
        while i < len(buffer):
            l = min(ls, len(buffer) - i)
            while True:
                p = buffer[i - ss : i + l - 1].rfind(buffer[i : i + l])
                if p >= 0:
                    break
                l -= 1
            if l >= break_even:  # Reference
                tokens.append((1, ss - p, l))
                i += l
                pbar.update(l)
            else:  # Literal
                tokens.append((0, buffer[i]))
                i += 1
                pbar.update(1)

        pbar.close()
        return tokens

    def tokens_to_source(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Decodes a given list of tokens to the corresponding sequence of source symbols.

        Examples:
            >>> lzss = komm.LempelZivSSCode(
            ...     search_size=8,
            ...     lookahead_size=4,
            ...     source_cardinality=4,
            ... )
            >>> lzss.tokens_to_source([(0, 3), (1, 4, 4), (0, 2), (1, 3, 2)])
            array([3, 0, 0, 0, 3, 2, 0, 3])
        """
        ss = self.search_size
        buffer = self.search_buffer.copy()

        for token in tokens:
            match token:
                case (1, p, l):  # Reference
                    for _ in range(l):
                        buffer.append(buffer[-p])
                case (0, x):  # Literal
                    buffer.append(x)
                case _:  # pyright: ignore[reportUnnecessaryComparison]
                    raise ValueError(f"invalid token: {token}")

        source = np.array(buffer[ss:], dtype=int)
        return source

    def tokens_to_target(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Returns the target alphabet representation corresponding to a given list of tokens.

        Examples:
            >>> lzss = komm.LempelZivSSCode(
            ...     search_size=8,
            ...     lookahead_size=4,
            ...     source_cardinality=4,
            ... )
            >>> lzss.tokens_to_target([(0, 3), (1, 4, 4), (0, 2), (1, 3, 2)])
            array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
        """
        calY = self.target_cardinality
        p_width, l_width, x_width = self._get_widths()
        target: list[int] = []

        for token in tokens:
            match token:
                case (1, p, l):  # Reference
                    target.append(1)
                    target.extend(integer_to_symbols(p - 1, base=calY, width=p_width))
                    target.extend(integer_to_symbols(l - 1, base=calY, width=l_width))
                case (0, x):  # Literal
                    target.append(0)
                    target.extend(integer_to_symbols(x, base=calY, width=x_width))
                case _:  # pyright: ignore[reportUnnecessaryComparison]
                    raise ValueError(f"invalid token: {token}")

        return np.array(target, dtype=int)

    def target_to_tokens(self, target: npt.ArrayLike) -> list[Token]:
        r"""
        Returns the list of tokens corresponding to a given target alphabet representation.

        Examples:
            >>> lzss = komm.LempelZivSSCode(
            ...     search_size=8,
            ...     lookahead_size=4,
            ...     source_cardinality=4,
            ... )
            >>> lzss.target_to_tokens([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
            [(0, 3), (1, 4, 4), (0, 2), (1, 3, 2)]
        """
        target = np.asarray(target, dtype=int)
        calY = self.target_cardinality
        p_width, l_width, x_width = self._get_widths()
        tokens: list[Token] = []
        i = 0

        while i < target.size:
            flag = target[i]
            i += 1
            if flag == 1:  # Reference
                if i + p_width + l_width <= target.size:
                    p = int(symbols_to_integer(target[i : i + p_width], base=calY)) + 1
                    i += p_width
                    l = int(symbols_to_integer(target[i : i + l_width], base=calY)) + 1
                    i += l_width
                    tokens.append((1, p, l))
            else:  # Literal
                if i + x_width <= target.size:
                    x = int(symbols_to_integer(target[i : i + x_width], base=calY))
                    i += x_width
                    tokens.append((0, x))

        return tokens

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a sequence of source symbols to a sequence of target symbols.

        Parameters:
            input: The sequence of source symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded target symbols. It is a 1D-array with elements in $\mathcal{Y}$.

        Examples:
            >>> lzss = komm.LempelZivSSCode(
            ...     search_size=8,
            ...     lookahead_size=4,
            ...     source_cardinality=4,
            ... )
            >>> lzss.encode([3, 0, 0, 0, 3, 2, 0, 3])
            array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
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
            >>> lzss = komm.LempelZivSSCode(
            ...     search_size=8,
            ...     lookahead_size=4,
            ...     source_cardinality=4,
            ... )
            >>> lzss.decode([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
            array([3, 0, 0, 0, 3, 2, 0, 3])
        """
        tokens = self.target_to_tokens(input)
        output = self.tokens_to_source(tokens)
        return output
