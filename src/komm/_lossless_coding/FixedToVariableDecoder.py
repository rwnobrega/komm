import numpy as np
import numpy.typing as npt
from attrs import frozen

from .FixedToVariableCode import FixedToVariableCode
from .util import parse_prefix_free


@frozen
class FixedToVariableDecoder:
    r"""
    Decoder for prefix-free [fixed-to-variable length codes](/ref/FixedToVariableCode).

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
        ValueError: code is not prefix-free
    """

    code: FixedToVariableCode

    def __attrs_post_init__(self) -> None:
        if not self.code.is_prefix_free():
            raise ValueError("code is not prefix-free")

    def __call__(self, in0: npt.ArrayLike) -> npt.NDArray[np.int_]:
        in0 = np.asarray(in0)
        out0 = parse_prefix_free(in0, self.code.inv_enc_mapping)
        return out0
