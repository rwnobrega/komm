from functools import cache
from operator import itemgetter

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .. import abc
from .._util.docs import mkdocstrings
from .._util.validators import validate_pmf
from ..types import Array1D
from .util import Word


@mkdocstrings(filters=["!.*"])
class FanoCode(abc.SymbolCode):
    r"""
    Binary Fano code. For a given pmf $p$ over $\mathcal{X}$, it is a [symbol code](/ref/SymbolCode) in which the source symbols are first sorted in descending order of probability and then are recursively partitioned into two groups of approximately equal total probability, assigning bit $\mathtt{0}$ to one group and bit $\mathtt{1}$ to the other, until each source symbol is assigned a unique codeword. For more details, see [Wikipedia: Shannon–Fano coding](https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano_coding).

    Notes:
        Fano codes are always [prefix-free](/ref/SymbolCode/#is_prefix_free) (hence [uniquely decodable](/ref/SymbolCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be non-negative and sum to $1$.

    Examples:
        >>> pmf = [0.2, 0.4, 0.2, 0.1, 0.1]

        >>> code = komm.FanoCode(pmf)
        >>> code.codewords
        [(1, 0), (0,), (1, 1, 0), (1, 1, 1, 0), (1, 1, 1, 1)]
        >>> code.average_length(pmf)  # doctest: +FLOAT_CMP
        2.2
    """

    def __init__(self, pmf: npt.ArrayLike):
        self.pmf = validate_pmf(pmf)
        if not self.pmf.size >= 2:
            raise ValueError(f"'pmf' must have size at least 2 (got {self.pmf.size})")
        self._enc_mapping = fano_code(self.pmf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pmf={self.pmf.tolist()})"

    @property
    def source_cardinality(self) -> int:
        return self.pmf.size

    @property
    def target_cardinality(self) -> int:
        return 2

    @property
    def enc_mapping(self) -> dict[int, Word]:
        return self._enc_mapping

    @cache
    def is_uniquely_decodable(self) -> bool:
        return True

    @cache
    def is_prefix_free(self) -> bool:
        return True


def fano_code(pmf: Array1D[np.floating]) -> dict[int, Word]:
    pbar = tqdm(desc="Generating Fano code", total=pmf.size, delay=2.5)
    codewords: list[Word] = [()] * pmf.size
    items = sorted(enumerate(pmf), key=itemgetter(1), reverse=True)
    stack: list[tuple[int, int, Word]] = [(0, pmf.size, ())]
    while stack:
        lo, hi, word = stack.pop()
        if hi - lo == 1:
            x, _ = items[lo]
            codewords[x] = word
            pbar.update()
            continue
        probs = [p for _, p in items[lo:hi]]
        index = int(np.argmin(np.abs(np.cumsum(probs) - np.sum(probs) / 2)))
        mid = lo + index + 1
        stack.append((mid, hi, word + (1,)))
        stack.append((lo, mid, word + (0,)))
    pbar.close()
    return dict(enumerate(codewords))
