from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import product
from math import prod
from typing import Literal

import numpy.typing as npt
from tqdm import tqdm
from typing_extensions import Self

from .._util.information_theory import PMF
from .FixedToVariableCode import FixedToVariableCode
from .util import Word


def HuffmanCode(
    pmf: npt.ArrayLike,
    source_block_size: int = 1,
    policy: Literal["high", "low"] = "high",
) -> FixedToVariableCode:
    r"""
    Binary Huffman code. It is an optimal (minimal expected rate) [fixed-to-variable length code](/ref/FixedToVariableCode) for a given probability mass function. For more details, see <cite>Say06, Sec. 3.2</cite>.

    Notes:
        Huffman codes are always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The probability mass function of the source.
        source_block_size: The source block size $k$. The default value is $k = 1$.
        policy: The policy to be used when constructing the code. It must be either `'high'` (move combined symbols as high as possible) or `'low'` (move combined symbols as low as possible). The default value is `'high'`.

    Examples:
        >>> pmf = [0.7, 0.15, 0.15]

        >>> code = komm.HuffmanCode(pmf)
        >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
        {(0,): (0,),
         (1,): (1, 1),
         (2,): (1, 0)}
        >>> code.rate(pmf)  # doctest: +NUMBER
        np.float64(1.3)

        >>> code = komm.HuffmanCode(pmf, 2)
        >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
        {(0, 0): (1,),
         (0, 1): (0, 0, 0, 0),
         (0, 2): (0, 1, 1),
         (1, 0): (0, 1, 0),
         (1, 1): (0, 0, 0, 1, 1, 1),
         (1, 2): (0, 0, 0, 1, 1, 0),
         (2, 0): (0, 0, 1),
         (2, 1): (0, 0, 0, 1, 0, 1),
         (2, 2): (0, 0, 0, 1, 0, 0)}
        >>> code.rate(pmf)  # doctest: +NUMBER
        np.float64(1.1975)
    """
    pmf = PMF(pmf)
    if not policy in {"high", "low"}:
        raise ValueError("'policy': must be in {'high', 'low'}")
    return FixedToVariableCode.from_codewords(
        source_cardinality=pmf.size,
        codewords=huffman_algorithm(pmf, source_block_size, policy),
    )


def huffman_algorithm(
    pmf: PMF, source_block_size: int, policy: Literal["high", "low"]
) -> list[Word]:
    @dataclass
    class Node:
        index: int
        probability: float
        parent: int | None = None
        bit: int = -1

        def __lt__(self, other: Self) -> bool:
            i0, p0 = self.index, self.probability
            i1, p1 = other.index, other.probability
            if policy == "high":
                return (p0, i0) < (p1, i1)
            elif policy == "low":
                return (p0, -i0) < (p1, -i1)

    pbar = tqdm(
        desc="Generating Huffman code",
        total=3 * pmf.size**source_block_size - 1,
        delay=2.5,
    )

    tree: list[Node] = []
    for index, probs in enumerate(product(pmf, repeat=source_block_size)):
        tree.append(Node(index, prod(probs)))
        pbar.update()

    heap = tree.copy()
    heapify(heap)
    while len(heap) > 1:
        node1 = heappop(heap)
        node0 = heappop(heap)
        node1.bit = 1
        node0.bit = 0
        node = Node(index=len(tree), probability=node0.probability + node1.probability)
        node0.parent = node1.parent = node.index
        heappush(heap, node)
        tree.append(node)
        pbar.update()

    codewords: list[Word] = []
    for index in range(pmf.size**source_block_size):
        node = tree[index]
        bits: list[int] = []
        while node.parent is not None:
            bits.append(node.bit)
            node = tree[node.parent]
        codewords.append(tuple(reversed(bits)))
        pbar.update()

    pbar.close()

    return codewords
