from itertools import chain

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.progress import tqdm_bar
from .._util.validators import (
    validate_integer,
    validate_integer_array,
    validate_positive_integer,
)
from ..types import Array1D
from .util import (
    ceil_log,
    find_longest_match,
    integer_to_symbols,
    symbols_to_integer,
    take,
)

Token = tuple[int, int, int]


class LempelZiv77Code(abc.TokenCode[Token]):
    r"""
    Lempel–Ziv 77 (LZ77 or LZ1) code. It is a lossless data compression algorithm based on a sliding window. For more details, see the [original LZ77 paper](https://doi.org/10.1109%2FTIT.1977.1055714), <cite>Say06, Sec. 5.4.1</cite> and <cite>CT06, Sec. 13.4.1</cite>.

    Let $\mathcal{X}$ be the source alphabet, $S \geq 1$ be the size of the *search buffer*, and $L \geq 1$ be the size of the *lookahead buffer*; the *sliding window* has size $W = S + L$. The source sequence is parsed into *phrases*, one per token. The token format is $(p, \ell, x)$, where $p \in [1 : S]$ is the *pointer* (location of the beginning of the match, measured backward from the end of the search buffer), $\ell \in [0 : L)$ is the *length* of the match, and $x \in \mathcal{X}$ is the source symbol following the match. The phrase of a token is the match followed by $x$, with $k = \ell + 1$ symbols.

    Let $\mathcal{Y}$ be the target alphabet. A token is represented as a word in $\mathcal{Y}^n$, consisting of $p - 1$ followed by $\ell$ and $x$, where $$n = \lceil \log S \rceil + \lceil \log L \rceil + \lceil \log |\mathcal{X}| \rceil$$ and all logs are to base $|\mathcal{Y}|$. Integers are written most significant digit first.

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
        self.search_size = validate_positive_integer(search_size)
        self.lookahead_size = validate_positive_integer(lookahead_size)
        self.source_cardinality = validate_integer(source_cardinality, low=2)
        self.target_cardinality = validate_integer(target_cardinality, low=2)
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
        p_width = ceil_log(self.search_size, calY)
        l_width = ceil_log(self.lookahead_size, calY)
        x_width = ceil_log(self.source_cardinality, calY)
        return p_width, l_width, x_width

    def source_to_tokens(self, source: npt.ArrayLike) -> list[Token]:
        r"""
        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.source_to_tokens([0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2])
            [(1, 2, 1), (2, 3, 2), (3, 7, 2)]
        """
        source = validate_integer_array(
            source, low=0, high=self.source_cardinality, ndim=1
        )
        ss, ls = self.search_size, self.lookahead_size
        buffer = bytes(self.search_buffer + source.tolist())
        tokens: list[Token] = []
        pbar = tqdm_bar(len(buffer), "Compressing LZ77", initial=ss)
        i = ss
        while i < len(buffer):
            p, l = find_longest_match(buffer, i, ss, min(ls, len(buffer) - i) - 1)
            tokens.append((ss - p, l, buffer[i + l]))
            i += l + 1
            pbar.update(l + 1)
        pbar.close()
        return tokens

    def tokens_to_source(self, tokens: list[Token]) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.tokens_to_source([(1, 2, 1), (2, 3, 2), (3, 7, 2)])
            array([0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2])
        """
        ss, ls, calX = self.search_size, self.lookahead_size, self.source_cardinality
        buffer = self.search_buffer.copy()
        for p, l, x in tokens:
            if not (1 <= p <= ss and 0 <= l < ls and 0 <= x < calX):
                raise ValueError(f"invalid token: {(p, l, x)}")
            start = len(buffer) - p
            for j in range(l):
                buffer.append(buffer[start + j])
            buffer.append(x)
        source = np.array(buffer[ss:], dtype=int)
        return source

    def tokens_to_target(self, tokens: list[Token]) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.tokens_to_target([(1, 2, 1), (2, 3, 2), (3, 7, 2)])
            array([0, 0, 0, 2, 1, 0, 1, 1, 0, 2, 0, 2, 2, 1, 2])
        """
        ss, ls, calX = self.search_size, self.lookahead_size, self.source_cardinality
        calY = self.target_cardinality
        p_width, l_width, x_width = self._get_widths()
        target: list[int] = []
        for p, l, x in tokens:
            if not (1 <= p <= ss and 0 <= l < ls and 0 <= x < calX):
                raise ValueError(f"invalid token: {(p, l, x)}")
            target.extend(integer_to_symbols(p - 1, base=calY, width=p_width))
            target.extend(integer_to_symbols(l, base=calY, width=l_width))
            target.extend(integer_to_symbols(x, base=calY, width=x_width))
        return np.array(target, dtype=int)

    def target_to_tokens(self, target: npt.ArrayLike) -> list[Token]:
        r"""
        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.target_to_tokens([0, 0, 0, 2, 1, 0, 1, 1, 0, 2, 0, 2, 2, 1, 2])
            [(1, 2, 1), (2, 3, 2), (3, 7, 2)]
        """
        target = validate_integer_array(
            target, low=0, high=self.target_cardinality, ndim=1
        )
        calY = self.target_cardinality
        p_width, l_width, x_width = self._get_widths()
        tokens: list[Token] = []
        it = iter(target.tolist())
        for first in it:
            symbols = chain([first], it)
            p = symbols_to_integer(take(symbols, p_width), base=calY) + 1
            l = symbols_to_integer(take(symbols, l_width), base=calY)
            x = symbols_to_integer(take(symbols, x_width), base=calY)
            tokens.append((p, l, x))
        return tokens

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.encode([0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2])
            array([0, 0, 0, 2, 1, 0, 1, 1, 0, 2, 0, 2, 2, 1, 2])
        """
        return super().encode(input)

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        r"""
        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     search_size=9,
            ...     lookahead_size=9,
            ...     source_cardinality=3,
            ...     target_cardinality=3,
            ... )
            >>> lz77.decode([0, 0, 0, 2, 1, 0, 1, 1, 0, 2, 0, 2, 2, 1, 2])
            array([0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2])
        """
        return super().decode(input)
