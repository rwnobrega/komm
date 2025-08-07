from functools import cache, reduce
from math import ceil, log2
from operator import itemgetter

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.docs import mkdocstrings
from .._util.validators import validate_pmf
from ..types import Array1D
from .FixedToVariableCode import FixedToVariableCode
from .util import Word


@mkdocstrings(filters=["!.*"])
class ShannonCode(FixedToVariableCode):
    r"""
    Binary Shannon code. For a given pmf $p$ over $\mathcal{X}$, it is a [fixed-to-variable length code](/ref/FixedToVariableCode) in which the length of the codeword $\Enc(\mathbf{x})$ associated with a source word $\mathbf{x} \in \mathcal{X}^k$ is given by
    $$
        \ell(\mathbf{x}) = \left\lceil \log_2 \frac{1}{p(\mathbf{x})} \right\rceil.
    $$
    This function implements the lexicographic order assignment as described in [Wikipedia: Shannon–Fano coding](https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano_coding).

    Notes:
        Shannon codes are always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be non-negative and sum to $1$.

        source_block_size: The source block size $k$. The default value is $k = 1$.

    Examples:
        >>> pmf = [0.8, 0.1, 0.1]

        >>> code = komm.ShannonCode(pmf)
        >>> code.enc_mapping
        {(0,): (0,),
         (1,): (1, 0, 0, 0),
         (2,): (1, 0, 0, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(1.6)

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
        np.float64(1.1)
    """

    def __init__(self, pmf: npt.ArrayLike, source_block_size: int = 1):
        self.pmf = validate_pmf(pmf)
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


def next_in_lexicographic_order(word: Word) -> Word:
    word_list = list(word)
    for i in range(len(word_list) - 1, -1, -1):
        if word_list[i] == 0:
            word_list[i] = 1
            break
        word_list[i] = 0
    return tuple(word_list)


def shannon_code(pmf: Array1D[np.floating], source_block_size: int) -> dict[Word, Word]:
    extended_pmf = reduce(np.multiply.outer, [pmf] * source_block_size)
    pbar = tqdm(desc="Generating Shannon code", total=extended_pmf.size, delay=2.5)
    enc_mapping: dict[Word, Word] = {x: () for x in np.ndindex(extended_pmf.shape)}
    y = ()
    for x, px in sorted(np.ndenumerate(extended_pmf), key=itemgetter(1), reverse=True):
        if px > 0:
            length = max(ceil(log2(1 / px)), 1)
            y = next_in_lexicographic_order(y) + (0,) * (length - len(y))
            enc_mapping[x] = y
        pbar.update()
    pbar.close()
    return enc_mapping
