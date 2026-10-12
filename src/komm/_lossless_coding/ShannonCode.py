from functools import cache

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.docs import mkdocstrings
from .._util.validators import validate_pmf
from ..types import Array1D
from .util import Word, canonical_code


@mkdocstrings(filters=["!.*"])
class ShannonCode(abc.SymbolCode):
    r"""
    Binary Shannon code. For a given pmf $p$ over $\mathcal{X}$, it is a [symbol code](/ref/SymbolCode) in which the length of the codeword $\Enc(x)$ associated with a source symbol $x \in \mathcal{X}$ is given by
    $$
        \ell(x) = \left\lceil \log_2 \frac{1}{p(x)} \right\rceil.
    $$
    For more details, see [Wikipedia: Shannon–Fano coding](https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano_coding).

    Notes:
        Shannon codes are always [prefix-free](/ref/SymbolCode/#is_prefix_free) (hence [uniquely decodable](/ref/SymbolCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be positive and sum to $1$. Zero probabilities are not allowed, since $\ell(x)$ is undefined for $p(x) = 0$.

    Examples:
        >>> pmf = [0.2, 0.4, 0.2, 0.1, 0.1]

        >>> code = komm.ShannonCode(pmf)
        >>> code.codewords
        [(0, 1, 0), (0, 0), (0, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1)]
        >>> code.average_length(pmf)  # doctest: +FLOAT_CMP
        2.8
    """

    def __init__(self, pmf: npt.ArrayLike):
        self.pmf = validate_pmf(pmf)
        if not self.pmf.size >= 2:
            raise ValueError(f"'pmf' must have size at least 2 (got {self.pmf.size})")
        if not np.all(self.pmf > 0):
            raise ValueError("'pmf' must be positive")
        self._enc_mapping = shannon_code(self.pmf)

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


def shannon_code_lengths(pmf: Array1D[np.floating]) -> Array1D[np.integer]:
    # The lower bound of 1 handles the degenerate case p(x) = 1.
    lengths = np.maximum(1, np.ceil(np.log2(1 / pmf)))
    return lengths.astype(int)


def shannon_code(pmf: Array1D[np.floating]) -> dict[int, Word]:
    lengths = shannon_code_lengths(pmf)
    codewords = canonical_code(lengths)
    return dict(enumerate(codewords))
