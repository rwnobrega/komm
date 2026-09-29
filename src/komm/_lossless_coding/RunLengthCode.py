from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_integer_range
from .util import integer_to_symbols, symbols_to_integer

Token = tuple[int, int]


@dataclass
class RunLengthCode:
    r"""
    Run-length code. It is a lossless data compression algorithm which parses the source sequence into *runs*, that is, blocks of repeated symbols. Let $\mathcal{X}$ be the source alphabet, and $\mathcal{Y}$ be the target alphabet. The notation used here is the following: $L \geq 1$ is the *maximum run length*. The token format is $(x, \ell)$, where $x \in \mathcal{X}$ is the repeated symbol and $\ell \in [1 : L]$ is the length of the run; runs longer than $L$ are split into two or more tokens. A token is represented as a fixed-size word in $\mathcal{Y}^n$, consisting of $x$ followed by $\ell - 1$, where $$n = \log |\mathcal{X}| + \log L$$ and all logs are to base $|\mathcal{Y}|$. For more details, see [Wikipedia: Run-length encoding](https://en.wikipedia.org/wiki/Run-length_encoding).

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.

    Parameters:
        source_cardinality: The source cardinality $|\mathcal{X}|$. Must satisfy $|\mathcal{X}| \geq 2$.
        max_run_length: The maximum run length $L$. Must satisfy $L \geq 1$.
        target_cardinality: The target cardinality $|\mathcal{Y}|$. Must satisfy $|\mathcal{Y}| \geq 2$. The default value is $2$ (binary).

    Examples:
        >>> code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
    """

    source_cardinality: int
    max_run_length: int
    target_cardinality: int = 2

    def __post_init__(self) -> None:
        if not self.source_cardinality >= 2:
            raise ValueError("'source_cardinality' must be at least 2")
        if not self.max_run_length >= 1:
            raise ValueError("'max_run_length' must be at least 1")
        if not self.target_cardinality >= 2:
            raise ValueError("'target_cardinality' must be at least 2")

    def _get_widths(self) -> tuple[int, int]:
        calY = self.target_cardinality
        x_width = ceil(log(self.source_cardinality, calY))
        l_width = ceil(log(self.max_run_length, calY))
        return x_width, l_width

    def source_to_tokens(self, source: npt.ArrayLike) -> list[Token]:
        r"""
        Encodes a given sequence of source symbols to the corresponding list of tokens.

        Examples:
            >>> code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
            >>> code.source_to_tokens([0, 0, 0, 2, 2, 1, 1, 1, 1, 1])
            [(0, 3), (2, 2), (1, 4), (1, 1)]
        """
        source = validate_integer_range(source, high=self.source_cardinality)
        L = self.max_run_length
        starts = np.flatnonzero(np.diff(source, prepend=-1))
        lengths = np.diff(starts, append=source.size)
        tokens: list[Token] = []
        for x, l in zip(source[starts].tolist(), lengths.tolist()):
            q, r = divmod(l, L)
            tokens += [(x, L)] * q
            if r > 0:
                tokens.append((x, r))
        return tokens

    def tokens_to_source(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Decodes a given list of tokens to the corresponding sequence of source symbols.

        Examples:
            >>> code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
            >>> code.tokens_to_source([(0, 3), (2, 2), (1, 4), (1, 1)])
            array([0, 0, 0, 2, 2, 1, 1, 1, 1, 1])
        """
        symbols, lengths = np.array(tokens, dtype=int).reshape(-1, 2).T
        return np.repeat(symbols, lengths)

    def tokens_to_target(self, tokens: list[Token]) -> npt.NDArray[np.integer]:
        r"""
        Returns the target alphabet representation corresponding to a given list of tokens.

        Examples:
            >>> code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
            >>> code.tokens_to_target([(0, 3), (2, 2), (1, 4), (1, 1)])
            array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
        """
        calX, calY = self.source_cardinality, self.target_cardinality
        x_width, l_width = self._get_widths()
        target: list[int] = []
        for token in tokens:
            x, l = token
            if not (0 <= x < calX and 1 <= l <= self.max_run_length):
                raise ValueError(f"invalid token: {token}")
            target.extend(integer_to_symbols(x, base=calY, width=x_width))
            target.extend(integer_to_symbols(l - 1, base=calY, width=l_width))
        return np.array(target, dtype=int)

    def target_to_tokens(self, target: npt.ArrayLike) -> list[Token]:
        r"""
        Returns the list of tokens corresponding to a given target alphabet representation.

        Examples:
            >>> code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
            >>> code.target_to_tokens([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
            [(0, 3), (2, 2), (1, 4), (1, 1)]
        """
        target = np.asarray(target, dtype=int)
        calY = self.target_cardinality
        x_width, l_width = self._get_widths()
        tokens: list[Token] = []
        i = 0
        while i + x_width + l_width <= target.size:
            x = symbols_to_integer(target[i : i + x_width], base=calY)
            i += x_width
            l = symbols_to_integer(target[i : i + l_width], base=calY)
            i += l_width
            tokens.append((x, l + 1))
        return tokens

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a sequence of source symbols to a sequence of target symbols.

        Parameters:
            input: The sequence of source symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X}$.

        Returns:
            output: The sequence of encoded target symbols. It is a 1D-array with elements in $\mathcal{Y}$.

        Examples:
            >>> code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
            >>> code.encode([0, 0, 0, 2, 2, 1, 1, 1, 1, 1])
            array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
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
            >>> code = komm.RunLengthCode(source_cardinality=3, max_run_length=4)
            >>> code.decode([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
            array([0, 0, 0, 2, 2, 1, 1, 1, 1, 1])
        """
        tokens = self.target_to_tokens(input)
        output = self.tokens_to_source(tokens)
        return output
