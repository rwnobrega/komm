import heapq

import numpy as np

from .VariableToFixedCode import VariableToFixedCode


class TunstallCode(VariableToFixedCode):
    r"""
    Tunstall code. It is an optimal (minimal expected rate) [variable-to-fixed length code](/ref/VariableToFixedCode) for a given probability mass function.
    """

    def __init__(self, pmf, code_block_size):
        r"""
        Constructor for the class.

        Parameters:

            pmf (Array1D[float]): The probability mass function of the source.

            code_block_size (Optional[int]): The code block size $n$. Must satisfy $2^n \geq |\mathcal{X}|$, where $|\mathcal{X}|$ is the cardinality of the source alphabet, given by `len(pmf)`.

        Examples:

            >>> pmf = [0.7, 0.15, 0.15]

            >>> code = komm.TunstallCode(pmf, code_block_size=2)
            >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0,): (0, 0),
             (1,): (0, 1),
             (2,): (1, 0)}
            >>> np.around(code.rate(pmf), decimals=6)
            2.0

            >>> code = komm.TunstallCode(pmf, code_block_size=3)
            >>> code.enc_mapping  # doctest: +NORMALIZE_WHITESPACE
            {(0, 0, 0): (0, 0, 0),
             (0, 0, 1): (0, 0, 1),
             (0, 0, 2): (0, 1, 0),
             (0, 1): (0, 1, 1),
             (0, 2): (1, 0, 0),
             (1,): (1, 0, 1),
             (2,): (1, 1, 0)}
            >>> np.around(code.rate(pmf), decimals=6)
            1.369863
        """
        self._pmf = np.array(pmf)
        self._code_block_size = code_block_size

        if 2**self._code_block_size < self._pmf.size:
            raise ValueError("Code block size is too low")

        super().__init__(sourcewords=TunstallCode._tunstall_algorithm(self._pmf, self._code_block_size))

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

        while len(queue) + pmf.size - 1 < 2**code_block_size:
            node = heapq.heappop(queue)
            for symbol, probability in enumerate(pmf):
                new_node = Node(node.symbols + (symbol,), node.probability * probability)
                heapq.heappush(queue, new_node)
        sourcewords = sorted(node.symbols for node in queue)

        return sourcewords

    def __repr__(self):
        args = f"pmf={self._pmf.tolist()}, code_block_size={self._code_block_size}"
        return f"{self.__class__.__name__}({args})"
