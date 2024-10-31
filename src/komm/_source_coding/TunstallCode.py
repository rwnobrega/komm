import heapq

import numpy as np
from attrs import field

from .._validation import is_pmf, validate_call
from .VariableToFixedCode import VariableToFixedCode


@validate_call(
    pmf=field(converter=np.asarray, validator=is_pmf),
)
def TunstallCode(pmf, target_block_size=None):
    r"""
    Binary Tunstall code. It is an optimal (minimal expected rate) [variable-to-fixed length code](/ref/VariableToFixedCode) for a given probability mass function.

    Parameters:

        pmf (Array1D[float]): The probability mass function of the source.

        target_block_size (Optional[int]): The target block size $n$. Must satisfy $2^n \geq S$, where $S$ is the cardinality of the source alphabet, given by `len(pmf)`. The default value is $n = \lceil \log_2 S \rceil$.

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
    if target_block_size is None:
        target_block_size = np.ceil(np.log2(pmf.size)).astype(int)
    elif 2**target_block_size < pmf.size:
        raise ValueError("'target_block_size' is too low")
    return VariableToFixedCode.from_sourcewords(
        target_cardinality=2,
        sourcewords=tunstall_algorithm(pmf, target_block_size),
    )


def tunstall_algorithm(pmf, code_block_size):
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
