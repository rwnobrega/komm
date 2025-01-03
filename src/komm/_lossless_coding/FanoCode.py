import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.information_theory import PMF
from .FixedToVariableCode import FixedToVariableCode
from .util import Word, empty_mapping, extended_probabilities


def FanoCode(
    pmf: npt.ArrayLike,
    source_block_size: int = 1,
) -> FixedToVariableCode:
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
        >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
        {(0,): (0,),
         (1,): (1, 0),
         (2,): (1, 1)}
        >>> code.rate(pmf)  # doctest: +NUMBER
        np.float64(1.3)

        >>> code = komm.FanoCode(pmf, 2)
        >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
        {(0, 0): (0,),
         (0, 1): (1, 0, 0),
         (0, 2): (1, 0, 1),
         (1, 0): (1, 1, 0),
         (1, 1): (1, 1, 1, 1, 0, 0),
         (1, 2): (1, 1, 1, 1, 0, 1),
         (2, 0): (1, 1, 1, 0),
         (2, 1): (1, 1, 1, 1, 1, 0),
         (2, 2): (1, 1, 1, 1, 1, 1)}
        >>> code.rate(pmf)  # doctest: +NUMBER
        np.float64(1.1975)
    """
    pmf = PMF(pmf)
    return FixedToVariableCode(
        source_cardinality=pmf.size,
        target_cardinality=2,
        source_block_size=source_block_size,
        enc_mapping=fano_algorithm(pmf, source_block_size),
    )


def fano_algorithm(pmf: PMF, source_block_size: int) -> dict[Word, Word]:
    pbar = tqdm(
        desc="Generating Fano code",
        total=2 * pmf.size**source_block_size,
        delay=2.5,
    )

    enc_mapping = empty_mapping(pmf.size, source_block_size)
    xpmf = extended_probabilities(pmf, source_block_size, pbar)
    stack: list[tuple[list[tuple[Word, float]], Word]] = [(xpmf, ())]
    while stack:
        current_pmf, prefix = stack.pop()
        if len(current_pmf) == 1:
            u, _ = current_pmf[0]
            enc_mapping[u] = prefix
            pbar.update()
            continue
        probs = [p for _, p in current_pmf]
        total = np.sum(probs)
        index = np.argmin(np.abs(np.cumsum(probs) - total / 2))
        stack.append((current_pmf[index + 1 :], prefix + (1,)))
        stack.append((current_pmf[: index + 1], prefix + (0,)))

    pbar.close()

    return enc_mapping
