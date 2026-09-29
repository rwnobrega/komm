from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._util.validators import validate_integer_range

Token = tuple[int, int]


@dataclass
class RunLengthCode:
    r"""
    Run-length code. It is a lossless data compression algorithm which parses the source sequence into *runs*, that is, blocks of repeated symbols. Let $\mathcal{X}$ be the source alphabet, and $\mathcal{Y}$ be the target alphabet. The notation used here is the following: $L \geq 1$ is the *maximum run length*. The token format is $(x, \ell)$, where $x \in \mathcal{X}$ is the repeated symbol and $\ell \in [1 : L]$ is the length of the run; runs longer than $L$ are split into two or more tokens. For more details, see <cite>Say06, Sec. 7.6.1</cite>.

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
