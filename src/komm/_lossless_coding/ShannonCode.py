from fractions import Fraction
from math import floor
from operator import itemgetter
from typing import Literal

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .. import abc
from .._util.docs import mkdocstrings
from .._util.validators import validate_choice, validate_pmf
from ..types import Array1D
from .util import Word, canonical_code, integer_to_symbols


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
        pmf: The pmf $p$ to be considered. Must be a 1D-array of size $|\mathcal{X}|$ of positive values that sum to $1$. Zero probabilities are not allowed, since $\ell(x)$ is undefined for $p(x) = 0$.

        assignment: The strategy used to assign the codewords, once the lengths are computed. Must be either `'cumulative'` (the codeword of $x$ is formed by the first $\ell(x)$ bits of the binary expansion of the total probability of the source symbols that precede $x$ in descending order of probability, as in Shannon's original construction) or `'canonical'` (the [canonical code](/ref/SymbolCode/#from_lengths) with the same lengths). The default value is `'cumulative'`. It affects neither the codeword lengths nor the [average length](/ref/SymbolCode/#average_length).

    Examples:
        >>> pmf = [0.2, 0.4, 0.2, 0.1, 0.1]

        >>> code = komm.ShannonCode(pmf)
        >>> code.codewords
        [(0, 1, 1), (0, 0), (1, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0)]
        >>> code.average_length(pmf)  # doctest: +FLOAT_CMP
        2.8

        >>> code = komm.ShannonCode(pmf, assignment="canonical")
        >>> code.codewords
        [(0, 1, 0), (0, 0), (0, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1)]
        >>> code.average_length(pmf)  # doctest: +FLOAT_CMP
        2.8
    """

    def __init__(
        self,
        pmf: npt.ArrayLike,
        *,
        assignment: Literal["cumulative", "canonical"] = "cumulative",
    ):
        self.pmf = validate_pmf(pmf)
        if not self.pmf.size >= 2:
            raise ValueError(f"'pmf' must have size at least 2 (got {self.pmf.size})")
        if not np.all(self.pmf > 0):
            raise ValueError("'pmf' must be positive")
        assignment = validate_choice(assignment, ("cumulative", "canonical"))
        self.assignment = assignment
        self._enc_mapping = shannon_code(self.pmf, assignment)

    def __repr__(self) -> str:
        args = ", ".join([
            f"pmf={self.pmf.tolist()}",
            f"assignment={self.assignment!r}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    def source_cardinality(self) -> int:
        return self.pmf.size

    @property
    def target_cardinality(self) -> int:
        return 2

    @property
    def enc_mapping(self) -> dict[int, Word]:
        return self._enc_mapping


def shannon_code_lengths(pmf: Array1D[np.floating]) -> Array1D[np.integer]:
    # The lower bound of 1 handles the degenerate case p(x) = 1.
    lengths = np.maximum(1, np.ceil(np.log2(1 / pmf)))
    return lengths.astype(int)


def cumulative_code(
    pmf: Array1D[np.floating],
    lengths: Array1D[np.integer],
) -> list[Word]:
    pbar = tqdm(desc="Generating Shannon code", total=pmf.size, delay=2.5)
    # Exact sums, slack of epsilon per symbol.
    eps = Fraction(np.finfo(float).eps)
    slack = min(pmf.size * eps, Fraction(float(pmf.min())) / 2)
    codewords: list[Word] = [()] * pmf.size
    total = end = Fraction(0)
    for x, p in sorted(enumerate(pmf), key=itemgetter(1), reverse=True):
        length = int(lengths[x])
        # Clamp, since float sums may exceed one.
        value = min(floor((total + slack) * 2**length), 2**length - 1)
        if not value >= end * 2**length:
            raise ValueError("'pmf' must sum to 1.0")
        codewords[x] = integer_to_symbols(value, base=2, width=length)
        total += Fraction(float(p))
        end = Fraction(value + 1, 2**length)
        pbar.update()
    pbar.close()
    return codewords


def shannon_code(
    pmf: Array1D[np.floating],
    assignment: Literal["cumulative", "canonical"],
) -> dict[int, Word]:
    lengths = shannon_code_lengths(pmf)
    if assignment == "cumulative":
        codewords = cumulative_code(pmf, lengths)
    else:
        codewords = canonical_code(lengths)
    return dict(enumerate(codewords))
