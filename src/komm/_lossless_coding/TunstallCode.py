from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from math import ceil, log2

import numpy.typing as npt
from tqdm import tqdm
from typing_extensions import Self

from .._util.information_theory import PMF
from .util import Word
from .VariableToFixedCode import VariableToFixedCode


def TunstallCode(
    pmf: npt.ArrayLike,
    target_block_size: int | None = None,
) -> VariableToFixedCode:
    r"""
    Binary Tunstall code. It is an optimal (minimal expected rate) [variable-to-fixed length code](/ref/VariableToFixedCode) for a given probability mass function. For more details, see <cite>Say06, Sec. 3.7</cite>.

    Notes:
        Tunstall codes are always [prefix-free](/ref/VariableToFixedCode/#is_prefix_free) (hence [uniquely encodable](/ref/VariableToFixedCode/#is_uniquely_encodable)) and [fully covering](/ref/VariableToFixedCode/#is_fully_covering).

    Parameters:
        pmf: The probability mass function of the source.
        target_block_size: The target block size $n$. Must satisfy $2^n \geq S$, where $S$ is the cardinality of the source alphabet, given by `len(pmf)`. The default value is $n = \lceil \log_2 S \rceil$.

    Examples:
        >>> pmf = [0.7, 0.15, 0.15]

        >>> code = komm.TunstallCode(pmf)
        >>> code.dec_mapping  # doctest: +NORMALIZE_WHITESPACE
        {(0, 0): (0,),
         (0, 1): (1,),
         (1, 0): (2,)}
        >>> code.rate(pmf)  # doctest: +NUMBER
        np.float64(2.0)

        >>> code = komm.TunstallCode(pmf, 3)
        >>> code.dec_mapping  # doctest: +NORMALIZE_WHITESPACE
        {(0, 0, 0): (0, 0, 0),
         (0, 0, 1): (0, 0, 1),
         (0, 1, 0): (0, 0, 2),
         (0, 1, 1): (0, 1),
         (1, 0, 0): (0, 2),
         (1, 0, 1): (1,),
         (1, 1, 0): (2,)}
        >>> code.rate(pmf)  # doctest: +NUMBER
        np.float64(1.3698630137)
    """
    pmf = PMF(pmf)
    if target_block_size is None:
        target_block_size = ceil(log2(pmf.size))
    elif 2**target_block_size < pmf.size:
        raise ValueError("'target_block_size' is too low")
    return VariableToFixedCode.from_sourcewords(
        target_cardinality=2,
        sourcewords=tunstall_algorithm(pmf, target_block_size),
    )


def tunstall_algorithm(pmf: PMF, code_block_size: int) -> list[Word]:
    @dataclass
    class Node:
        sourceword: Word
        probability: float

        def __lt__(self, other: Self) -> bool:
            return -self.probability < -other.probability

    pbar = tqdm(
        desc="Generating Tunstall code",
        total=2 ** (code_block_size - 1) - pmf.size + 1,
        delay=2.5,
    )

    heap = [Node((symbol,), probability) for (symbol, probability) in enumerate(pmf)]
    heapify(heap)
    while len(heap) + pmf.size - 1 < 2**code_block_size:
        node = heappop(heap)
        for symbol, probability in enumerate(pmf):
            new_node = Node(node.sourceword + (symbol,), node.probability * probability)
            heappush(heap, new_node)
        pbar.update()

    pbar.close()

    return sorted(node.sourceword for node in heap)
