from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_block import BlockCode
from .._util import bits_to_int


@dataclass
class SyndromeTableDecoder(abc.BlockDecoder[BlockCode]):
    r"""
    Syndrome table decoder for general [block codes](/ref/BlockCode). This decoder implements syndrome-based hard-decision decoding using a precomputed table of coset leaders.

    Parameters:
        code: The block code to be used for decoding.

    Parameters: Input:
        r: The input received word(s). Can be a single received word of length $n$ or a multidimensional array where the last dimension has length $n$.

    Parameters: Output:
        u_hat: The output message(s). Has the same shape as the input, with the last dimension reduced from $n$ to $k$.

    Notes:
        - Input type: `hard`.
        - Output type: `hard`.

    Examples:
        >>> code = komm.HammingCode(3)
        >>> decoder = komm.SyndromeTableDecoder(code)
        >>> decoder([[1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0]])
        array([[1, 1, 0, 0],
               [1, 0, 1, 1]])
    """

    code: BlockCode

    def __post_init__(self) -> None:
        self.coset_leaders = self.code.coset_leaders()

    def _decode(self, r: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        s = self.code.check(r)
        e_hat = self.coset_leaders[bits_to_int(s)]
        v_hat = np.bitwise_xor(r, e_hat)
        u_hat = self.code.inverse_encode(v_hat)
        return u_hat
