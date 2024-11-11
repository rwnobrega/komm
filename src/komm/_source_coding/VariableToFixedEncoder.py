import numpy as np
import numpy.typing as npt
from attrs import frozen

from ._util import parse_prefix_free
from .VariableToFixedCode import VariableToFixedCode


@frozen
class VariableToFixedEncoder:
    r"""
    Encoder for prefix-free [variable-to-fixed length codes](/ref/VariableToFixedCode).

    Attributes:
        code: The code to be considered.

    Parameters: Input:
        in0 (Array1D[int]): The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$, where $S$ is the source cardinality of the code.

    Parameters: Output:
        out0 (Array1D[int]): The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$, where $T$ is the target cardinality of the code.

    Examples:
        >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (1,)])
        >>> encoder = komm.VariableToFixedEncoder(code)
        >>> encoder([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])

        >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (0,)])
        >>> encoder = komm.VariableToFixedEncoder(code)
        Traceback (most recent call last):
        ...
        ValueError: code is not prefix-free
    """

    code: VariableToFixedCode

    def __attrs_post_init__(self):
        if not self.code.is_prefix_free():
            raise ValueError("code is not prefix-free")

    def __call__(self, in0: npt.ArrayLike) -> np.ndarray:
        out0 = np.array(parse_prefix_free(in0, self.code.inv_dec_mapping))
        return out0
