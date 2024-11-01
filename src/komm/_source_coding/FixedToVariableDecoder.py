import numpy as np
import numpy.typing as npt
from attrs import mutable

from ._util import parse_prefix_free
from .FixedToVariableCode import FixedToVariableCode


@mutable
class FixedToVariableDecoder:
    r"""
    Prefix-free decoder for [fixed-to-variable length code](/ref/FixedToVariableCode).

    Attributes:
        code: The code to be considered, which must be a prefix-free code (that is, no codeword is a prefix of another codeword).

    Parameters: Input:
        in0 (Array1D[int]): The sequence of symbols to be decoded. Must be a 1D-array with elements in $[0:T)$, where $T$ is the target cardinality of the code.

    Parameters: Output:
        out0 (Array1D[int]): The sequence of decoded symbols. It is a 1D-array with elements in $[0:S)$, where $S$ is the source cardinality of the code.

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

    def __call__(self, in0: npt.ArrayLike) -> np.ndarray:
        out0 = np.array(parse_prefix_free(in0, self.code.inv_enc_mapping))
        return out0
