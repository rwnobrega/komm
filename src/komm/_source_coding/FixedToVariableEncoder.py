import numpy as np
from attrs import define

from .FixedToVariableCode import FixedToVariableCode


@define
class FixedToVariableEncoder:
    r"""
    Encoder for [fixed-to-variable length code](/ref/FixedToVariableCode).

    Attributes:

        code (FixedToVariableCode): The code to be considered.

    Examples:

        >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
        >>> encoder = komm.FixedToVariableEncoder(code)
        >>> encoder([1, 0, 1, 0, 2, 0])
        array([1, 0, 0, 1, 0, 0, 1, 1, 0])
    """
    code: FixedToVariableCode

    def __call__(self, x) -> np.ndarray:
        k, enc = self.code.source_block_size, self.code.enc_mapping
        return np.concatenate([enc[tuple(s)] for s in np.reshape(x, newshape=(-1, k))])
