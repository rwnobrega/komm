import numpy as np
import numpy.typing as npt
from attrs import mutable

from .FixedToVariableCode import FixedToVariableCode


@mutable
class FixedToVariableEncoder:
    r"""
    Encoder for [fixed-to-variable length code](/ref/FixedToVariableCode).

    Attributes:
        code: The code to be considered.

    Parameters: Input:
        in0 (Array1D[int]): The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$, where $S$ is the source cardinality of the code.

    Parameters: Output:
        out0 (Array1D[int]): The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$, where $T$ is the target cardinality of the code.

    Examples:
        >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
        >>> encoder = komm.FixedToVariableEncoder(code)
        >>> encoder([1, 0, 1, 0, 2, 0])
        array([1, 0, 0, 1, 0, 0, 1, 1, 0])
    """

    code: FixedToVariableCode

    def __call__(self, in0: npt.ArrayLike) -> np.ndarray:
        k, enc = self.code.source_block_size, self.code.enc_mapping
        out0 = np.concatenate([enc[tuple(s)] for s in np.reshape(in0, shape=(-1, k))])
        return out0
