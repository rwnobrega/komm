import heapq
import itertools as it
import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .._util.information_theory import PMF

Word = tuple[int, ...]


def is_prefix_free(words: list[Word]) -> bool:
    for c1, c2 in it.permutations(words, 2):
        if c1[: len(c2)] == c2:
            return False
    return True


def is_uniquely_decodable(words: list[Word]) -> bool:
    # Sardinas–Patterson algorithm. See [Say06, Sec. 2.4.1].
    augmented_words = set(words)
    while True:
        dangling_suffixes: set[Word] = set()
        for c1, c2 in it.permutations(augmented_words, 2):
            if c1[: len(c2)] == c2:
                dangling_suffixes.add(c1[len(c2) :])
        if dangling_suffixes & set(words):
            return False
        if dangling_suffixes <= augmented_words:
            return True
        augmented_words |= dangling_suffixes


def parse_prefix_free(
    input_sequence: npt.NDArray[np.integer], dictionary: dict[Word, Word]
) -> npt.NDArray[np.integer]:
    output_sequence: list[int] = []
    i = 0
    while i < len(input_sequence):
        j = 1
        while i + j <= len(input_sequence):
            try:
                key = tuple(input_sequence[i : i + j])
                output_sequence.extend(dictionary[key])
                break
            except KeyError:
                j += 1
        i += j
    return np.asarray(output_sequence)


def huffman_algorithm(
    pmf: PMF, source_block_size: int, policy: Literal["high", "low"]
) -> list[Word]:
    class Node:
        def __init__(self, index: int, probability: float):
            self.index: int = index
            self.probability: float = probability
            self.parent: int | None = None
            self.bit: int = -1

        def __lt__(self, other: Self) -> bool:
            i0, p0 = self.index, self.probability
            i1, p1 = other.index, other.probability
            if policy == "high":
                return (p0, i0) < (p1, i1)
            elif policy == "low":
                return (p0, -i0) < (p1, -i1)

    tree = [
        Node(i, math.prod(probs))
        for (i, probs) in enumerate(it.product(pmf, repeat=source_block_size))
    ]
    queue = [node for node in tree]
    heapq.heapify(queue)
    while len(queue) > 1:
        node1 = heapq.heappop(queue)
        node0 = heapq.heappop(queue)
        node1.bit = 1
        node0.bit = 0
        node = Node(index=len(tree), probability=node0.probability + node1.probability)
        node0.parent = node1.parent = node.index
        heapq.heappush(queue, node)
        tree.append(node)

    codewords: list[Word] = []
    for symbol in range(pmf.size**source_block_size):
        node = tree[symbol]
        bits: list[int] = []
        while node.parent is not None:
            bits.insert(0, node.bit)
            node = tree[node.parent]
        codewords.append(tuple(bits))

    return codewords


def tunstall_algorithm(pmf: PMF, code_block_size: int) -> list[Word]:
    class Node:
        def __init__(self, symbols: Word, probability: float):
            self.symbols = symbols
            self.probability = probability

        def __lt__(self, other: Self) -> bool:
            return -self.probability < -other.probability

    queue = [Node((symbol,), probability) for (symbol, probability) in enumerate(pmf)]
    heapq.heapify(queue)

    while len(queue) + pmf.size - 1 < 2**code_block_size:
        node = heapq.heappop(queue)
        for symbol, probability in enumerate(pmf):
            new_node = Node(node.symbols + (symbol,), node.probability * probability)
            heapq.heappush(queue, new_node)
    sourcewords = sorted(node.symbols for node in queue)

    return sourcewords
