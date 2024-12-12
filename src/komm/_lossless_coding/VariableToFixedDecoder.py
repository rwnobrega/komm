import numpy as np
import numpy.typing as npt
from attrs import frozen

from .VariableToFixedCode import VariableToFixedCode


@frozen
class VariableToFixedDecoder:
    r"""
    Decoder for [variable-to-fixed length codes](/ref/VariableToFixedCode).

    Attributes:
        code: The code to be considered.

    :::komm.VariableToFixedDecoder.VariableToFixedDecoder.__call__
    """

    code: VariableToFixedCode

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Parameters: Input:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $[0:T)$, where $T$ is the target cardinality of the code.

        Returns: Output:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $[0:S)$, where $S$ is the source cardinality of the code.

        Examples:
            >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (1,)])
            >>> decoder = komm.VariableToFixedDecoder(code)
            >>> decoder([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
            array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        """
        input = np.asarray(input)
        n, dec = self.code.target_block_size, self.code.dec_mapping
        output = np.concatenate([dec[tuple(s)] for s in input.reshape(-1, n)])
        return output
