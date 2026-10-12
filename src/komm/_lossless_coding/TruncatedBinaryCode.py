from collections.abc import Iterator
from functools import cached_property
from itertools import chain
from typing import SupportsIndex

import numpy as np
import numpy.typing as npt

from .. import abc
from .._integer_coding.base import take
from .._util.bit_operations import from_binary, to_binary
from .._util.docs import mkdocstrings
from .._util.validators import validate_integer, validate_integer_array
from ..types import Array1D
from .util import Word


@mkdocstrings(filters=["!.*"])
class TruncatedBinaryCode(abc.SymbolCode):
    r"""
    Truncated binary code. It is a binary [symbol code](/ref/SymbolCode) with source alphabet $\mathcal{X} = [0 : M)$, where $M \geq 2$ is a given cardinality. Let $k = \lfloor \log_2 M \rfloor$ and $u = 2^{k+1} - M$. The codeword for a source symbol $x \in \mathcal{X}$ is the $k$-bit binary representation of $x$, if $x < u$, or the $(k + 1)$-bit binary representation of $x + u$, otherwise. For more details, see [Wikipedia: Truncated binary encoding](https://en.wikipedia.org/wiki/Truncated_binary_encoding).

    The table below shows the codewords for small values of $M$.

    | $x$ | $M = 4$ | $M = 5$ | $M = 6$ | $M = 7$ | $M = 8$ |
    | :-: | ------- | ------- | ------- | ------- | ------- |
    | $0$ | `00`    | `00`    | `00`    | `00`    | `000`   |
    | $1$ | `01`    | `01`    | `01`    | `010`   | `001`   |
    | $2$ | `10`    | `10`    | `100`   | `011`   | `010`   |
    | $3$ | `11`    | `110`   | `101`   | `100`   | `011`   |
    | $4$ |         | `111`   | `110`   | `101`   | `100`   |
    | $5$ |         |         | `111`   | `110`   | `101`   |
    | $6$ |         |         |         | `111`   | `110`   |
    | $7$ |         |         |         |         | `111`   |

    Notes:
        - For $M$ a power of $2$ it reduces to the fixed-length binary code.
        - It coincides with the [Huffman code](/ref/HuffmanCode) for the uniform pmf, with canonical assignment.

    Parameters:
        cardinality: The cardinality $M$ of the code. Must satisfy $M \geq 2$.

    Examples:
        >>> code = komm.TruncatedBinaryCode(5)

        >>> code.encode([4, 1, 3])
        array([1, 1, 1, 0, 1, 1, 1, 0])

        >>> code.decode([1, 1, 1, 0, 1, 1, 1, 0])
        array([4, 1, 3])
    """

    def __init__(self, cardinality: SupportsIndex) -> None:
        self.cardinality = validate_integer(cardinality, low=2)
        self._k = self.cardinality.bit_length() - 1
        self._u = 2 ** (self._k + 1) - self.cardinality

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cardinality={self.cardinality})"

    @property
    def source_cardinality(self) -> int:
        return self.cardinality

    @property
    def target_cardinality(self) -> int:
        return 2

    @property
    def enc_mapping(self) -> dict[int, Word]:
        return self._enc_mapping

    @cached_property
    def _enc_mapping(self) -> dict[int, Word]:
        return {x: self.encode_single(x) for x in range(self.cardinality)}

    def encode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        input = validate_integer_array(input, low=0, high=self.cardinality, ndim=1)
        output = [y for x in input.tolist() for y in self.encode_single(x)]
        return np.array(output, dtype=int)

    def decode(self, input: npt.ArrayLike) -> Array1D[np.integer]:
        input = validate_integer_array(input, low=0, high=2, ndim=1)
        it = iter(input.tolist())
        output = [self.decode_single(chain([first], it)) for first in it]
        return np.array(output, dtype=int)

    def encode_single(self, symbol: SupportsIndex) -> Word:
        symbol = validate_integer(symbol, low=0, high=self.cardinality)
        u, k = self._u, self._k
        if symbol < u:
            return tuple(to_binary(symbol, width=k, bit_order="MSB-first"))
        return tuple(to_binary(symbol + u, width=k + 1, bit_order="MSB-first"))

    def decode_single(self, symbols: Iterator[SupportsIndex]) -> int:
        u, k = self._u, self._k
        x = from_binary(take(symbols, k), bit_order="MSB-first")
        if x < u:
            return x
        return 2 * x + from_binary(take(symbols, 1), bit_order="MSB-first") - u
