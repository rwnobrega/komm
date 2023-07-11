import numpy as np
from attrs import define

from .util import parse_prefix_free
from .VariableToFixedCode import VariableToFixedCode


@define
class VariableToFixedEncoder:
    r"""
    Prefix-free encoder for [variable-to-fixed length code](/ref/VariableToFixedCode).

    Attributes:

        code (VariableToFixedCode): The code to be considered.

    Examples:

        >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (1,)])
        >>> encoder = komm.VariableToFixedEncoder(code)
        >>> encoder([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])

        >>> code = komm.VariableToFixedCode.from_sourcewords(2, [(0,0,0), (0,0,1), (0,1), (0,)])
        >>> encoder = komm.VariableToFixedEncoder(code)
        Traceback (most recent call last):
        ...
        ValueError: The code is not prefix-free.
    """
    code: VariableToFixedCode

    def __attrs_post_init__(self):
        if not self.code.is_prefix_free():
            raise ValueError("The code is not prefix-free.")

    def __call__(self, x) -> np.ndarray:
        return np.array(parse_prefix_free(x, self.code.inv_dec_mapping))
