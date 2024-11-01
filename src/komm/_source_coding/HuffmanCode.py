import heapq
import itertools

import numpy as np
from attrs import field, validators

from .._validation import is_pmf, validate_call
from .FixedToVariableCode import FixedToVariableCode


@validate_call(
    pmf=field(converter=np.asarray, validator=is_pmf),
    source_block_size=field(validator=validators.ge(1)),
    policy=field(validator=validators.in_(["high", "low"])),
)
def HuffmanCode(pmf, source_block_size=1, policy="high"):
    r"""
    Binary Huffman code. It is an optimal (minimal expected rate) [fixed-to-variable length code](/ref/FixedToVariableCode) for a given probability mass function.

    Parameters:

        pmf (Array1D[float]): The probability mass function of the source.

        source_block_size (Optional[int]): The source block size $k$. The default value is $k = 1$.

        policy (Optional[str]): The policy to be used when constructing the code. It must be either `'high'` (move combined symbols as high as possible) or `'low'` (move combined symbols as low as possible). The default value is `'high'`.

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
    return FixedToVariableCode.from_codewords(
        source_cardinality=pmf.size,
        codewords=huffman_algorithm(pmf, source_block_size, policy),
    )


def huffman_algorithm(pmf, source_block_size, policy):
    class Node:
        def __init__(self, index, probability):
            self.index: int = index
            self.probability: float = probability
            self.parent: int | None = None
            self.bit: int | None = None

        def __lt__(self, other):
            if policy == "high":
                return (self.probability, self.index) < (other.probability, other.index)
            elif policy == "low":
                return (self.probability, -self.index) < (
                    other.probability,
                    -other.index,
                )

    tree = [
        Node(i, np.prod(probs))
        for (i, probs) in enumerate(itertools.product(pmf, repeat=source_block_size))
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

    codewords = []
    for symbol in range(pmf.size**source_block_size):
        node = tree[symbol]
        bits = []
        while node.parent is not None:
            bits.insert(0, node.bit)
            node = tree[node.parent]
        codewords.append(tuple(bits))

    return codewords
