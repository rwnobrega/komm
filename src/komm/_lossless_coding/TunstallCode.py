from dataclasses import dataclass
from functools import cache
from heapq import heapify, heappop, heappush
from itertools import product
from math import ceil, log2

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing_extensions import Self

from .._util.docs import mkdocstrings
from .._util.validators import validate_pmf
from ..types import Array1D
from .util import Word
from .VariableToFixedCode import VariableToFixedCode


@mkdocstrings(filters=["!.*"])
class TunstallCode(VariableToFixedCode):
    r"""
    Binary Tunstall code. It is an optimal (minimal expected rate) [variable-to-fixed length code](/ref/VariableToFixedCode) for a given pmf $p$ over $\mathcal{X}$. For more details, see <cite>Say06, Sec. 3.7</cite>.

    Notes:
        Tunstall codes are always [prefix-free](/ref/VariableToFixedCode/#is_prefix_free) (hence [uniquely encodable](/ref/VariableToFixedCode/#is_uniquely_encodable)) and [fully covering](/ref/VariableToFixedCode/#is_fully_covering).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be non-negative and sum to $1$.

        target_block_size: The target block size $n$. Must satisfy $2^n \geq |\mathcal{X}|$. The default value is $n = \lceil \log_2 |\mathcal{X}| \rceil$.

    Examples:
        >>> pmf = [0.8, 0.1, 0.1]

        >>> code = komm.TunstallCode(pmf)
        >>> code.dec_mapping
        {(0, 0): (0,),
         (0, 1): (1,),
         (1, 0): (2,)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(2.0)

        >>> code = komm.TunstallCode(pmf, 3)
        >>> code.dec_mapping
        {(0, 0, 0): (0, 0, 0),
         (0, 0, 1): (0, 0, 1),
         (0, 1, 0): (0, 0, 2),
         (0, 1, 1): (0, 1),
         (1, 0, 0): (0, 2),
         (1, 0, 1): (1,),
         (1, 1, 0): (2,)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(1.2295081967213108)
    """

    def __init__(
        self,
        pmf: npt.ArrayLike,
        target_block_size: int | None = None,
    ) -> None:
        self.pmf = validate_pmf(pmf)
        if target_block_size is None:
            target_block_size = ceil(log2(self.pmf.size))
        if 2**target_block_size < self.pmf.size:
            raise ValueError("'target_block_size' is too low")
        super().__init__(
            target_cardinality=2,
            source_cardinality=self.pmf.size,
            target_block_size=target_block_size,
            dec_mapping=tunstall_code(self.pmf, target_block_size),
        )

    def __repr__(self) -> str:
        args = ", ".join([
            f"pmf={self.pmf.tolist()}",
            f"target_block_size={self.target_block_size}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cache
    def is_fully_covering(self) -> bool:
        return True

    @cache
    def is_uniquely_encodable(self) -> bool:
        return True

    @cache
    def is_prefix_free(self) -> bool:
        return True


def tunstall_code(
    pmf: Array1D[np.floating],
    target_block_size: int,
) -> dict[Word, Word]:

    @dataclass
    class Node:
        sourceword: Word
        probability: np.floating

        def __lt__(self, other: Self) -> np.bool:
            return -self.probability < -other.probability

    S, n = pmf.size, target_block_size
    K = ceil((2**n - S) / (S - 1)) - 1  # See [Say06, p. 70]
    size = S + K * (S - 1)

    pbar = tqdm(desc="Generating Tunstall code", total=K * S + size, delay=2.5)

    heap = [Node((symbol,), probability) for (symbol, probability) in enumerate(pmf)]
    heapify(heap)
    for _ in range(K):
        node = heappop(heap)
        for symbol, probability in enumerate(pmf):
            new_node = Node(node.sourceword + (symbol,), node.probability * probability)
            heappush(heap, new_node)
            pbar.update()

    dec_mapping: dict[Word, Word] = {}
    for y, x in zip(
        product([0, 1], repeat=n), sorted(node.sourceword for node in heap)
    ):
        dec_mapping[y] = x
        pbar.update()

    pbar.close()

    return dec_mapping
