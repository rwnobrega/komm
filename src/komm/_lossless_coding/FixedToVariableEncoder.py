import numpy as np
import numpy.typing as npt
from attrs import frozen

from .FixedToVariableCode import FixedToVariableCode


@frozen
class FixedToVariableEncoder:
    r"""
    Encoder for [fixed-to-variable length codes](/ref/FixedToVariableCode).

    Attributes:
        code: The code to be considered.

    :::komm.FixedToVariableEncoder.FixedToVariableEncoder.__call__
    """

    code: FixedToVariableCode

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Parameters: Input:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$, where $S$ is the source cardinality of the code.

        Returns: Output:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$, where $T$ is the target cardinality of the code.

        Examples:
            >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
            >>> encoder = komm.FixedToVariableEncoder(code)
            >>> encoder([1, 0, 1, 0, 2, 0])
            array([1, 0, 0, 1, 0, 0, 1, 1, 0])
        """
        input = np.asarray(input)
        k, enc = self.code.source_block_size, self.code.enc_mapping
        output = np.concatenate([enc[tuple(s)] for s in input.reshape(-1, k)])
        return output
