import numpy as np
from attrs import define

from .FixedToVariableCode import FixedToVariableCode
from .util import parse_prefix_free


@define
class FixedToVariableDecoder:
    r"""
    Prefix-free decoder for [fixed-to-variable length code](/ref/FixedToVariableCode).

    Attributes:

        code (FixedToVariableCode): The code to be considered, which must be a prefix-free code (that is, no codeword is a prefix of another codeword).

    Examples:

        >>> code = komm.FixedToVariableCode.from_codewords(3, [(0,), (1,0), (1,1)])
        >>> decoder = komm.FixedToVariableDecoder(code)
        >>> decoder([1, 0, 0, 1, 0, 0, 1, 1, 0])
        array([1, 0, 1, 0, 2, 0])

        >>> code = komm.FixedToVariableCode.from_codewords(2, [(0,), (1,0), (1,1), (1,1,0)])
        >>> decoder = komm.FixedToVariableDecoder(code)
        Traceback (most recent call last):
        ...
        ValueError: The code is not prefix-free.
    """
    code: FixedToVariableCode

    def __attrs_post_init__(self):
        if not self.code.is_prefix_free():
            raise ValueError("The code is not prefix-free.")

    def __call__(self, x) -> np.ndarray:
        return np.array(parse_prefix_free(x, self.code.inv_enc_mapping))
