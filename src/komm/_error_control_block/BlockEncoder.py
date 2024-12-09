import numpy as np
from attrs import frozen
from numpy import typing as npt

from .BlockCode import BlockCode


@frozen
class BlockEncoder:
    r"""
    Encoder for [linear block codes](/ref/BlockCode).

    Attributes:
        code: The [block code](/ref/BlockCode) to be considered.

    Parameters: Input:
        in0 (Array1D[int]): The bit sequence to be encoded. Its length must be a multiple of the code's dimension $k$.

    Parameters: Output:
        out0 (Array1D[int]): The encoded bit sequence. Its length is a multiple of the code's block length $n$.

    Examples:
        >>> code = komm.HammingCode(3)
        >>> encoder = komm.BlockEncoder(code)
        >>> encoder([1, 1, 0, 0, 1, 0, 1, 1])
        array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0])
    """

    code: BlockCode

    def __call__(self, in0: npt.ArrayLike) -> npt.NDArray[np.integer]:
        u = np.reshape(in0, (-1, self.code.dimension))
        v = np.apply_along_axis(self.code.enc_mapping, 1, u)
        out0 = np.reshape(v, (-1,))
        return out0
