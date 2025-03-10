from functools import cache

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.docs import mkdocstrings
from .._util.information_theory import PMF
from .FixedToVariableCode import FixedToVariableCode
from .util import Word, empty_mapping, extended_probabilities


@mkdocstrings(filters=["!.*"])
class FanoCode(FixedToVariableCode):
    r"""
    Binary Fano code. It is a [fixed-to-variable length code](/ref/FixedToVariableCode) in which the source words are first sorted in descending order of probability and then are recursively partitioned into two groups of approximately equal total probability, assigning bit $\mathtt{0}$ to one group and bit $\mathtt{1}$ to the other, until each source word is assigned a unique codeword. For more details, see [Wikipedia: Shannon–Fano coding](https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano_coding).

    Notes:
        Fano codes are always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The probability mass function of the source.
        source_block_size: The source block size $k$. The default value is $k = 1$.

    Examples:
        >>> pmf = [0.7, 0.15, 0.15]

        >>> code = komm.FanoCode(pmf, 1)
        >>> code.enc_mapping
        {(0,): (0,),
         (1,): (1, 0),
         (2,): (1, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(1.3)

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
        np.float64(1.1975)
    """

    def __init__(self, pmf: npt.ArrayLike, source_block_size: int = 1):
        self.pmf = PMF(pmf)
        super().__init__(
            source_cardinality=self.pmf.size,
            target_cardinality=2,
            source_block_size=source_block_size,
            enc_mapping=fano_algorithm(self.pmf, source_block_size),
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


def fano_algorithm(pmf: PMF, source_block_size: int) -> dict[Word, Word]:
    pbar = tqdm(
        desc="Generating Fano code",
        total=2 * pmf.size**source_block_size,
        delay=2.5,
    )

    enc_mapping = empty_mapping(pmf.size, source_block_size)
    group = extended_probabilities(pmf, source_block_size, pbar)
    stack: list[tuple[list[tuple[Word, float]], Word]] = [(group, ())]
    while stack:
        group, v = stack.pop()
        if len(group) == 1:
            u, _ = group[0]
            enc_mapping[u] = v
            pbar.update()
            continue
        probs = [p for _, p in group]
        total = np.sum(probs)
        index = np.argmin(np.abs(np.cumsum(probs) - total / 2))
        stack.append((group[index + 1 :], v + (1,)))
        stack.append((group[: index + 1], v + (0,)))

    pbar.close()

    return enc_mapping
