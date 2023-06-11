import heapq
import itertools

import numpy as np

from .FixedToVariableCode import FixedToVariableCode


class HuffmanCode(FixedToVariableCode):
    r"""
    Huffman code. It is an optimal (minimal expected rate) [fixed-to-variable length code](/ref/FixedToVariableCode) for a given probability mass function.

    Examples:

        >>> code = komm.HuffmanCode([0.7, 0.15, 0.15])
        >>> pprint(code.enc_mapping)
        {(0,): (0,), (1,): (1, 1), (2,): (1, 0)}

        >>> code = komm.HuffmanCode([0.7, 0.15, 0.15], source_block_size=2)
        >>> pprint(code.enc_mapping)
        {(0, 0): (1,),
         (0, 1): (0, 0, 0, 0),
         (0, 2): (0, 1, 1),
         (1, 0): (0, 1, 0),
         (1, 1): (0, 0, 0, 1, 1, 1),
         (1, 2): (0, 0, 0, 1, 1, 0),
         (2, 0): (0, 0, 1),
         (2, 1): (0, 0, 0, 1, 0, 1),
         (2, 2): (0, 0, 0, 1, 0, 0)}
    """

    def __init__(self, pmf, source_block_size=1, policy="high"):
        r"""
        Constructor for the class.

        Parameters:

            pmf (1D-array of :obj:`float`): The probability mass function used to construct the code.

            source_block_size (:obj:`int`, optional): The source block size $k$. The default value is $k = 1$.

            policy (:obj:`str`, optional): The policy to be used when constructing the code. It must be either `'high'` (move combined symbols as high as possible) or `'low'` (move combined symbols as low as possible). The default value is `'high'`.
        """
        self._pmf = np.array(pmf)
        self._policy = policy

        if policy not in ["high", "low"]:
            raise ValueError("Parameter 'policy' must be in {'high', 'low'}")

        super().__init__(
            codewords=HuffmanCode._huffman_algorithm(pmf, source_block_size, policy), source_cardinality=self._pmf.size
        )

    @property
    def pmf(self):
        r"""
        The probability mass function used to construct the code.
        """
        return self._pmf

    @staticmethod
    def _huffman_algorithm(pmf, source_block_size, policy):
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
                    return (self.probability, -self.index) < (other.probability, -other.index)

        tree = [Node(i, np.prod(probs)) for (i, probs) in enumerate(itertools.product(pmf, repeat=source_block_size))]
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
        for symbol in range(len(pmf) ** source_block_size):
            node = tree[symbol]
            bits = []
            while node.parent is not None:
                bits.insert(0, node.bit)
                node = tree[node.parent]
            codewords.append(tuple(bits))

        return codewords

    def __repr__(self):
        args = "pmf={}, source_block_size={}".format(self._pmf.tolist(), self._source_block_size)
        return "{}({})".format(self.__class__.__name__, args)
