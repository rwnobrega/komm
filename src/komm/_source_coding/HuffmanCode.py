from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from attrs import field, validators

from .._validation import is_pmf, validate_call
from ._util import huffman_algorithm
from .FixedToVariableCode import FixedToVariableCode


@validate_call(
    pmf=field(converter=np.asarray, validator=is_pmf),
    source_block_size=field(validator=validators.ge(1)),
    policy=field(validator=validators.in_(["high", "low"])),
)
def HuffmanCode(
    pmf: npt.ArrayLike,
    source_block_size: int = 1,
    policy: Literal["high", "low"] = "high",
) -> FixedToVariableCode:
    r"""
    Binary Huffman code. It is an optimal (minimal expected rate) [fixed-to-variable length code](/ref/FixedToVariableCode) for a given probability mass function.

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
    pmf = cast(npt.NDArray[np.float64], pmf)
    return FixedToVariableCode.from_codewords(
        source_cardinality=pmf.size,
        codewords=huffman_algorithm(pmf, source_block_size, policy),
    )
