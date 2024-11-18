from typing import Optional, cast

import numpy as np
import numpy.typing as npt
from attrs import field

from komm._lossless_coding.util import tunstall_algorithm

from .._validation import is_pmf, validate_call
from .VariableToFixedCode import VariableToFixedCode


@validate_call(
    pmf=field(converter=np.asarray, validator=is_pmf),
)
def TunstallCode(
    pmf: npt.ArrayLike,
    target_block_size: Optional[int] = None,
) -> VariableToFixedCode:
    r"""
    Binary Tunstall code. It is an optimal (minimal expected rate) [variable-to-fixed length code](/ref/VariableToFixedCode) for a given probability mass function. For more details, see <cite>Say06, Sec. 3.7</cite>.

    Parameters:
        pmf: The probability mass function of the source.
        target_block_size: The target block size $n$. Must satisfy $2^n \geq S$, where $S$ is the cardinality of the source alphabet, given by `len(pmf)`. The default value is $n = \lceil \log_2 S \rceil$.

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
    pmf = cast(npt.NDArray[np.float64], pmf)
    if target_block_size is None:
        target_block_size = int(np.ceil(np.log2(pmf.size)))
    elif 2**target_block_size < pmf.size:
        raise ValueError("'target_block_size' is too low")
    return VariableToFixedCode.from_sourcewords(
        target_cardinality=2,
        sourcewords=tunstall_algorithm(pmf, target_block_size),
    )
