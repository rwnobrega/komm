from dataclasses import dataclass
from functools import cache, reduce
from heapq import heapify, heappop, heappush
from typing import Literal

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing_extensions import Self

from .._util.docs import mkdocstrings
from .._util.validators import validate_pmf
from ..types import Array1D
from .FixedToVariableCode import FixedToVariableCode
from .util import Word


@mkdocstrings(filters=["!.*"])
class HuffmanCode(FixedToVariableCode):
    r"""
    Binary Huffman code. It is an optimal (minimal expected rate) [fixed-to-variable length code](/ref/FixedToVariableCode) for a given pmf $p$ over $\mathcal{X}$. For more details, see <cite>Say06, Sec. 3.2</cite>.

    Notes:
        Huffman codes are always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be non-negative and sum to $1$.

        source_block_size: The source block size $k$. The default value is $k = 1$.

        policy: The policy to be used when constructing the code. It must be either `'high'` (move combined symbols as high as possible) or `'low'` (move combined symbols as low as possible). The default value is `'high'`.

    Examples:
        >>> pmf = [0.8, 0.1, 0.1]

        >>> code = komm.HuffmanCode(pmf)
        >>> code.enc_mapping
        {(0,): (0,),
         (1,): (1, 0),
         (2,): (1, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(1.2)

        >>> code = komm.HuffmanCode(pmf, 2)
        >>> code.enc_mapping
        {(0, 0): (0,),
         (0, 1): (1, 0, 1),
         (0, 2): (1, 1, 0),
         (1, 0): (1, 1, 1),
         (1, 1): (1, 0, 0, 1, 0, 0),
         (1, 2): (1, 0, 0, 1, 0, 1),
         (2, 0): (1, 0, 0, 0),
         (2, 1): (1, 0, 0, 1, 1, 0),
         (2, 2): (1, 0, 0, 1, 1, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(0.96)
    """

    def __init__(
        self,
        pmf: npt.ArrayLike,
        source_block_size: int = 1,
        policy: Literal["high", "low"] = "high",
    ):
        self.pmf = validate_pmf(pmf)
        if not source_block_size >= 1:
            raise ValueError("'source_block_size' must be at least 1")
        if not policy in {"high", "low"}:
            raise ValueError("'policy' must be in {'high', 'low'}")
        self.policy = policy
        super().__init__(
            source_cardinality=self.pmf.size,
            target_cardinality=2,
            source_block_size=source_block_size,
            enc_mapping=huffman_code(self.pmf, source_block_size, policy),
        )

    def __repr__(self) -> str:
        args = ", ".join([
            f"pmf={self.pmf.tolist()}",
            f"source_block_size={self.source_block_size}",
            f"policy={self.policy!r}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cache
    def is_uniquely_decodable(self) -> bool:
        return True

    @cache
    def is_prefix_free(self) -> bool:
        return True


def huffman_code(
    pmf: Array1D[np.floating],
    source_block_size: int,
    policy: Literal["high", "low"],
) -> dict[Word, Word]:

    @dataclass(slots=True)
    class Node:
        index: int
        probability: np.floating
        leaf: bool = True
        parent: int = -1
        bit: int = -1

        def __lt__(self, other: Self) -> bool:
            p0, i0 = self.probability, self.index
            p1, i1 = other.probability, other.index
            if policy == "high":
                return (p0, -i0 if self.leaf else i0) < (p1, -i1 if other.leaf else i1)
            elif policy == "low":
                return (p0, -i0) < (p1, -i1)

    extended_pmf = reduce(np.multiply.outer, [pmf] * source_block_size)

    pbar = tqdm(desc="Generating Huffman code", total=3 * extended_pmf.size, delay=2.5)
    pbar.update()

    tree: list[Node] = []
    for index, prob in enumerate(extended_pmf.ravel()):
        tree.append(Node(index, prob))
        pbar.update()

    heap = tree.copy()
    heapify(heap)
    while len(heap) > 1:
        node1 = heappop(heap)
        node0 = heappop(heap)
        node = Node(
            index=len(tree),
            probability=node0.probability + node1.probability,
            leaf=False,
        )
        node1.bit = 1
        node0.bit = 0
        node0.parent = node1.parent = node.index
        heappush(heap, node)
        tree.append(node)
        pbar.update()

    enc_mapping: dict[Word, Word] = {x: () for x in np.ndindex(extended_pmf.shape)}
    for index, x in enumerate(enc_mapping.keys()):
        node = tree[index]
        bits: list[int] = []
        while node.parent >= 0:
            bits.append(node.bit)
            node = tree[node.parent]
        y = tuple(reversed(bits))
        enc_mapping[x] = y
        pbar.update()

    pbar.close()

    return enc_mapping
