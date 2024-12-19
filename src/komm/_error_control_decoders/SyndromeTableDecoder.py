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

    Notes:
        - Input type: `hard`.
        - Output type: `hard`.

    # `__call__`

    :::komm.abc.BlockDecoder.BlockDecoder.__call__

    Examples:
        >>> code = komm.HammingCode(3)
        >>> decoder = komm.SyndromeTableDecoder(code)
        >>> decoder([[1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0]])
        array([[1, 1, 0, 0],
               [1, 0, 1, 1]])
    """

    code: BlockCode

    def __post_init__(self) -> None:
        self._coset_leaders = self.code.coset_leaders()

    def _decode(self, r: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        s = self.code.check(r)
        e_hat = self._coset_leaders[bits_to_int(s)]
        v_hat = np.bitwise_xor(r, e_hat)
        u_hat = self.code.inverse_encode(v_hat)
        return u_hat
