import numpy as np
from attrs import define

from .VariableToFixedCode import VariableToFixedCode


@define
class VariableToFixedDecoder:
    r"""
    Decoder for [variable-to-fixed length code](/ref/VariableToFixedCode).

    Attributes:

        code: The code to be considered.

    Examples:

        >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (1,)])
        >>> decoder = komm.VariableToFixedDecoder(code)
        >>> decoder([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
        array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    """
    code: VariableToFixedCode

    def __call__(self, x) -> np.ndarray:
        n, dec = self.code.target_block_size, self.code.dec_mapping
        return np.concatenate([dec[tuple(s)] for s in np.reshape(x, newshape=(-1, n))])
