from functools import cache, reduce

import numpy as np
import numpy.typing as npt

from .._util.docs import mkdocstrings
from .._util.validators import validate_pmf
from ..types import Array1D
from .FixedToVariableCode import FixedToVariableCode
from .util import Word, canonical_code


@mkdocstrings(filters=["!.*"])
class ShannonCode(FixedToVariableCode):
    r"""
    Binary Shannon code. For a given pmf $p$ over $\mathcal{X}$, it is a [fixed-to-variable length code](/ref/FixedToVariableCode) in which the length of the codeword $\Enc(\mathbf{x})$ associated with a source word $\mathbf{x} \in \mathcal{X}^k$ is given by
    $$
        \ell(\mathbf{x}) = \left\lceil \log_2 \frac{1}{p(\mathbf{x})} \right\rceil.
    $$
    This function implements the canonical (lexicographical) order assignment as described in [Wikipedia: Shannon–Fano coding](https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano_coding).

    Notes:
        Shannon codes are always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be positive and sum to $1$. Zero probabilities are not allowed, since $\ell(\mathbf{x})$ is undefined for $p(\mathbf{x}) = 0$.

        source_block_size: The source block size $k$. The default value is $k = 1$.

    Examples:
        >>> pmf = [0.8, 0.1, 0.1]

        >>> code = komm.ShannonCode(pmf)
        >>> code.enc_mapping
        {(0,): (0,),
         (1,): (1, 0, 0, 0),
         (2,): (1, 0, 0, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        1.6

        >>> code = komm.ShannonCode(pmf, 2)
        >>> code.enc_mapping
        {(0, 0): (0,),
         (0, 1): (1, 0, 0, 0),
         (0, 2): (1, 0, 0, 1),
         (1, 0): (1, 0, 1, 0),
         (1, 1): (1, 1, 0, 0, 0, 0, 0),
         (1, 2): (1, 1, 0, 0, 0, 0, 1),
         (2, 0): (1, 0, 1, 1),
         (2, 1): (1, 1, 0, 0, 0, 1, 0),
         (2, 2): (1, 1, 0, 0, 0, 1, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        1.1
    """

    def __init__(self, pmf: npt.ArrayLike, source_block_size: int = 1):
        self.pmf = validate_pmf(pmf)
        if not np.all(self.pmf > 0):
            raise ValueError("pmf must be positive")
        if not source_block_size >= 1:
            raise ValueError("'source_block_size' must be at least 1")
        super().__init__(
            source_cardinality=self.pmf.size,
            target_cardinality=2,
            source_block_size=source_block_size,
            enc_mapping=shannon_code(self.pmf, source_block_size),
        )

    def __repr__(self) -> str:
        args = ", ".join([
            f"pmf={self.pmf.tolist()}",
            f"source_block_size={self.source_block_size}",
        ])
        return f"{self.__class__.__name__}({args})"

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


def shannon_code(pmf: Array1D[np.floating], source_block_size: int) -> dict[Word, Word]:
    extended_pmf = reduce(np.multiply.outer, [pmf] * source_block_size)
    lengths = shannon_code_lengths(extended_pmf.ravel())
    codewords = canonical_code(lengths)
    enc_mapping: dict[Word, Word] = {}
    for x, c in zip(np.ndindex(extended_pmf.shape), codewords):
        enc_mapping[x] = c
    return enc_mapping
