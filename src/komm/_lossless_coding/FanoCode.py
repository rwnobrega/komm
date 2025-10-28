from functools import cache
from operator import itemgetter

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.docs import mkdocstrings
from .._util.validators import validate_pmf
from ..types import Array1D
from .FixedToVariableCode import FixedToVariableCode
from .util import Word, create_code_from_lengths


@mkdocstrings(filters=["!.*"])
class FanoCode(FixedToVariableCode):
    r"""
    Binary Fano code. For a given pmf $p$ over $\mathcal{X}$, it is a [fixed-to-variable length code](/ref/FixedToVariableCode) in which the source words are first sorted in descending order of probability and then are recursively partitioned into two groups of approximately equal total probability, assigning bit $\mathtt{0}$ to one group and bit $\mathtt{1}$ to the other, until each source word is assigned a unique codeword. For more details, see [Wikipedia: Shannon–Fano coding](https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano_coding).

    Notes:
        Fano codes are always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be non-negative and sum to $1$.

        source_block_size: The source block size $k$. The default value is $k = 1$.

    Examples:
        >>> pmf = [0.8, 0.1, 0.1]

        >>> code = komm.FanoCode(pmf)
        >>> code.enc_mapping
        {(0,): (0,),
         (1,): (1, 0),
         (2,): (1, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(1.2)

        >>> code = komm.FanoCode(pmf, 2)
        >>> code.enc_mapping
        {(0, 0): (0,),
         (0, 1): (1, 0, 0),
         (0, 2): (1, 0, 1),
         (1, 0): (1, 1, 0),
         (1, 1): (1, 1, 1, 1, 0, 0),
         (1, 2): (1, 1, 1, 1, 0, 1),
         (2, 0): (1, 1, 1, 0),
         (2, 1): (1, 1, 1, 1, 1, 0),
         (2, 2): (1, 1, 1, 1, 1, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(0.96)
    """

    def __init__(self, pmf: npt.ArrayLike, source_block_size: int = 1):
        self.pmf = validate_pmf(pmf)
        if not source_block_size >= 1:
            raise ValueError("'source_block_size' must be at least 1")
        super().__init__(
            source_cardinality=self.pmf.size,
            target_cardinality=2,
            source_block_size=source_block_size,
            enc_mapping=create_code_from_lengths(
                self.pmf, source_block_size, fano_code_lengths
            ),
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


def fano_code_lengths(pmf: Array1D[np.floating]) -> Array1D[np.integer]:
    pbar = tqdm(desc="Generating Fano code lengths", total=pmf.size, delay=2.5)
    lengths = np.zeros_like(pmf, dtype=int)
    items = sorted(np.ndenumerate(pmf), key=itemgetter(1), reverse=True)
    stack: list[tuple[int, int, int]] = [(0, pmf.size, 0)]
    while stack:
        lo, hi, length = stack.pop()
        if hi - lo == 1:
            x, _ = items[lo]
            lengths[x] = length
            pbar.update()
            continue
        probs = [p for _, p in items[lo:hi]]
        index = int(np.argmin(np.abs(np.cumsum(probs) - np.sum(probs) / 2)))
        mid = lo + index + 1
        stack.append((mid, hi, length + 1))
        stack.append((lo, mid, length + 1))
    pbar.close()
    return lengths
