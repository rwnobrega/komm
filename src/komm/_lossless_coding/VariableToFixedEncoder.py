import numpy as np
import numpy.typing as npt
from attrs import frozen

from .util import parse_prefix_free
from .VariableToFixedCode import VariableToFixedCode


@frozen
class VariableToFixedEncoder:
    r"""
    Encoder for prefix-free [variable-to-fixed length codes](/ref/VariableToFixedCode).

    Attributes:
        code: The code to be considered.

    :::komm.VariableToFixedEncoder.VariableToFixedEncoder.__call__
    """

    code: VariableToFixedCode

    def __attrs_post_init__(self) -> None:
        if not self.code.is_prefix_free():
            raise ValueError("code is not prefix-free")

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""

        Parameters: Input:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$, where $S$ is the source cardinality of the code.

        Returns: Output:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$, where $T$ is the target cardinality of the code.

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
        input = np.asarray(input)
        output = parse_prefix_free(input, self.code.inv_dec_mapping)
        return output
