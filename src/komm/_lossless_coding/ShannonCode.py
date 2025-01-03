from math import ceil, log2

import numpy.typing as npt
from tqdm import tqdm

from .._util.information_theory import PMF
from .FixedToVariableCode import FixedToVariableCode
from .util import Word, empty_mapping, extended_probabilities


def ShannonCode(
    pmf: npt.ArrayLike,
    source_block_size: int = 1,
) -> FixedToVariableCode:
    r"""
    Binary Shannon code. It is a [fixed-to-variable length code](/ref/FixedToVariableCode) in which the length of the codeword $\Enc(u)$ for a source symbol $u \in \mathcal{S}^k$ is given by
    $$
        \ell_u = \left\lceil \log_2 \frac{1}{p_u} \right\rceil,
    $$
    where $p_u$ is the probability of the source symbol $u$. This function implements the lexicographic order assignment as described in [Wikipedia: Shannon–Fano coding](https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano_coding).

    Notes:
        Shannon codes are always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The probability mass function of the source.
        source_block_size: The source block size $k$. The default value is $k = 1$.

    Examples:
        >>> pmf = [0.7, 0.15, 0.15]

        >>> code = komm.ShannonCode(pmf, 1)
        >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
        {(0,): (0,),
         (1,): (1, 0, 0),
         (2,): (1, 0, 1)}
        >>> code.rate(pmf)  # doctest: +NUMBER
        np.float64(1.6)

        >>> code = komm.ShannonCode(pmf, 2)
        >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
        {(0, 0): (0, 0),
         (0, 1): (0, 1, 0, 0),
         (0, 2): (0, 1, 0, 1),
         (1, 0): (0, 1, 1, 0),
         (1, 1): (1, 0, 0, 0, 0, 0),
         (1, 2): (1, 0, 0, 0, 0, 1),
         (2, 0): (0, 1, 1, 1),
         (2, 1): (1, 0, 0, 0, 1, 0),
         (2, 2): (1, 0, 0, 0, 1, 1)}
        >>> code.rate(pmf)  # doctest: +NUMBER
        np.float64(1.6)
    """
    pmf = PMF(pmf)
    return FixedToVariableCode(
        source_cardinality=pmf.size,
        target_cardinality=2,
        source_block_size=source_block_size,
        enc_mapping=shannon_code(pmf, source_block_size),
    )


def next_in_lexicographic_order(word: Word) -> Word:
    word_list = list(word)
    for i in range(len(word_list) - 1, -1, -1):
        if word_list[i] == 0:
            word_list[i] = 1
            break
        word_list[i] = 0
    return tuple(word_list)


def shannon_code(pmf: PMF, source_block_size: int) -> dict[Word, Word]:
    pbar = tqdm(
        desc="Generating Shannon code",
        total=2 * pmf.size**source_block_size,
        delay=2.5,
    )

    enc_mapping = empty_mapping(pmf.size, source_block_size)
    v = ()
    for u, pu in extended_probabilities(pmf, source_block_size, pbar):
        length = ceil(log2(1 / pu))
        v = next_in_lexicographic_order(v) + (0,) * (length - len(v))
        enc_mapping[u] = v
        pbar.update()

    pbar.close()

    return enc_mapping
