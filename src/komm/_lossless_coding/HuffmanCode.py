from dataclasses import dataclass
from functools import cache
from heapq import heapify, heappop, heappush
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .. import abc
from .._util.docs import mkdocstrings
from .._util.validators import validate_choice, validate_pmf
from ..types import Array1D
from .util import Word, canonical_code


@mkdocstrings(filters=["!.*"])
class HuffmanCode(abc.SymbolCode):
    r"""
    Binary Huffman code. It is an optimal (minimal average length) [symbol code](/ref/SymbolCode) for a given pmf $p$ over $\mathcal{X}$. For more details, see <cite>Say06, Sec. 3.2</cite>.

    Notes:
        Huffman codes are always [prefix-free](/ref/SymbolCode/#is_prefix_free) (hence [uniquely decodable](/ref/SymbolCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be non-negative and sum to $1$.

        policy: The policy to be used when constructing the code. It must be either `'high'` (move combined symbols as high as possible) or `'low'` (move combined symbols as low as possible). The default value is `'high'`. It affects the codeword lengths only in the presence of ties.

        assignment: The strategy used to assign the codewords, once the Huffman tree is built. It must be either `'tree'` (read the codewords off the paths of the tree, bit $\mathtt{1}$ being assigned to the less probable of the two merged nodes, as in classic textbook presentations) or `'canonical'` (the [canonical code](/ref/SymbolCode/#from_lengths) with the resulting lengths). The default value is `'tree'`. It affects neither the codeword lengths nor the [average length](/ref/SymbolCode/#average_length).

    Examples:
        >>> pmf = [0.2, 0.4, 0.2, 0.1, 0.1]

        >>> code = komm.HuffmanCode(pmf)
        >>> code.codewords
        [(1, 0), (0, 0), (1, 1), (0, 1, 0), (0, 1, 1)]
        >>> code.average_length(pmf)  # doctest: +FLOAT_CMP
        2.2

        >>> code = komm.HuffmanCode(pmf, policy="low")
        >>> code.codewords
        [(0, 1), (1,), (0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 1)]
        >>> code.average_length(pmf)  # doctest: +FLOAT_CMP
        2.2

        >>> code = komm.HuffmanCode(pmf, assignment="canonical")
        >>> code.codewords
        [(0, 0), (0, 1), (1, 0), (1, 1, 0), (1, 1, 1)]
        >>> code.average_length(pmf)  # doctest: +FLOAT_CMP
        2.2
    """

    def __init__(
        self,
        pmf: npt.ArrayLike,
        policy: Literal["high", "low"] = "high",
        assignment: Literal["tree", "canonical"] = "tree",
    ):
        self.pmf = validate_pmf(pmf)
        if not self.pmf.size >= 2:
            raise ValueError(f"'pmf' must have size at least 2 (got {self.pmf.size})")
        policy = validate_choice(policy, ("high", "low"))
        assignment = validate_choice(assignment, ("tree", "canonical"))
        self.policy = policy
        self.assignment = assignment
        self._enc_mapping = huffman_code(self.pmf, policy, assignment)

    def __repr__(self) -> str:
        args = ", ".join([
            f"pmf={self.pmf.tolist()}",
            f"policy={self.policy!r}",
            f"assignment={self.assignment!r}",
        ])
        return f"{self.__class__.__name__}({args})"

    @property
    def source_cardinality(self) -> int:
        return self.pmf.size

    @property
    def target_cardinality(self) -> int:
        return 2

    @property
    def enc_mapping(self) -> dict[int, Word]:
        return self._enc_mapping

    @cache
    def is_uniquely_decodable(self) -> bool:
        return True

    @cache
    def is_prefix_free(self) -> bool:
        return True


@dataclass(slots=True)
class Node:
    index: int
    probability: np.floating
    key: tuple[np.floating, int]
    leaf: bool = True
    parent: int = -1
    bit: int = -1

    def __lt__(self, other: Self) -> bool:
        return self.key < other.key


def huffman_tree(
    pmf: Array1D[np.floating],
    policy: Literal["high", "low"],
    pbar: "tqdm[Any]",
) -> list[Node]:
    def node(index: int, probability: np.floating, leaf: bool) -> Node:
        sign = 1 if policy == "high" and not leaf else -1
        return Node(index, probability, (probability, sign * index), leaf)

    tree: list[Node] = []
    for index, probability in enumerate(pmf):
        tree.append(node(index, probability, leaf=True))
        pbar.update()

    heap = tree.copy()
    heapify(heap)
    while len(heap) > 1:
        node1 = heappop(heap)
        node0 = heappop(heap)
        parent = node(
            index=len(tree),
            probability=node0.probability + node1.probability,
            leaf=False,
        )
        node1.bit = 1
        node0.bit = 0
        node0.parent = node1.parent = parent.index
        heappush(heap, parent)
        tree.append(parent)
        pbar.update()

    return tree


def tree_lengths(tree: list[Node], size: int, pbar: "tqdm[Any]") -> list[int]:
    depths = [0] * len(tree)
    for node in reversed(tree):
        if node.parent >= 0:
            depths[node.index] = depths[node.parent] + 1
        pbar.update(node.leaf)
    return depths[:size]


def tree_codewords(tree: list[Node], size: int, pbar: "tqdm[Any]") -> list[Word]:
    codewords: list[Word] = []
    for index in range(size):
        node = tree[index]
        bits: list[int] = []
        while node.parent >= 0:
            bits.append(node.bit)
            node = tree[node.parent]
        codewords.append(tuple(reversed(bits)))
        pbar.update()
    return codewords


def huffman_code(
    pmf: Array1D[np.floating],
    policy: Literal["high", "low"],
    assignment: Literal["tree", "canonical"],
) -> dict[int, Word]:
    size = pmf.size

    pbar = tqdm(desc="Generating Huffman code", total=3 * size, delay=2.5)
    tree = huffman_tree(pmf, policy, pbar)
    if assignment == "canonical":
        codewords = canonical_code(tree_lengths(tree, size, pbar))
    else:
        codewords = tree_codewords(tree, size, pbar)
    pbar.close()

    return dict(enumerate(codewords))
