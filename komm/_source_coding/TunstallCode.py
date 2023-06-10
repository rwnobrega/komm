import heapq

import numpy as np

from .VariableToFixedCode import VariableToFixedCode


class TunstallCode(VariableToFixedCode):
    r"""
    Tunstall code. It is an optimal (minimal expected rate) variable-to-fixed length code (:class:`VariableToFixedCode`) for a given probability mass function.

    Examples:

        >>> code = komm.TunstallCode([0.6, 0.3, 0.1], code_block_size=3)
        >>> pprint(code.enc_mapping)
        {(0, 0, 0): (0, 0, 0),
         (0, 0, 1): (0, 0, 1),
         (0, 0, 2): (0, 1, 0),
         (0, 1): (0, 1, 1),
         (0, 2): (1, 0, 0),
         (1,): (1, 0, 1),
         (2,): (1, 1, 0)}
    """

    def __init__(self, pmf, code_block_size):
        r"""
        Constructor for the class.

        Parameters:

            pmf (1D-array of :obj:`float`): The probability mass function used to construct the code.

            code_block_size (:obj:`int`, optional): The code block size $n$. Must satisfy $2^n \geq |\mathcal{X}|$, where $|\mathcal{X}|$ is the cardinality of the source alphabet, given by :code:`len(pmf)`.
        """
        self._pmf = np.array(pmf)

        if 2**code_block_size < len(pmf):
            raise ValueError("Code block size is too low")

        super().__init__(sourcewords=TunstallCode._tunstall_algorithm(pmf, code_block_size))

    @property
    def pmf(self):
        r"""
        The probability mass function used to construct the code. This property is read-only.
        """
        return self._pmf

    @staticmethod
    def _tunstall_algorithm(pmf, code_block_size):
        class Node:
            def __init__(self, symbols, probability):
                self.symbols = symbols
                self.probability = probability

            def __lt__(self, other):
                return -self.probability < -other.probability

        queue = [Node((symbol,), probability) for (symbol, probability) in enumerate(pmf)]
        heapq.heapify(queue)

        while len(queue) + len(pmf) - 1 < 2**code_block_size:
            node = heapq.heappop(queue)
            for symbol, probability in enumerate(pmf):
                new_node = Node(node.symbols + (symbol,), node.probability * probability)
                heapq.heappush(queue, new_node)
        sourcewords = sorted(node.symbols for node in queue)

        return sourcewords

    def __repr__(self):
        args = "pmf={}, code_block_size={}".format(self._pmf.tolist(), self._code_block_size)
        return "{}({})".format(self.__class__.__name__, args)
