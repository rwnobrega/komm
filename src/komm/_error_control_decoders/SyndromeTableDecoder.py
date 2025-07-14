from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.bit_operations import bits_to_int
from .._util.decorators import blockwise


@dataclass
class SyndromeTableDecoder(abc.BlockDecoder[abc.BlockCode]):
    r"""
    Syndrome table decoder for general [block codes](/ref/BlockCode). This decoder implements syndrome-based hard-decision decoding using a precomputed table of coset leaders.

    Parameters:
        code: The block code to be used for decoding.

    Notes:
        - Input type: `hard` (bits).
        - Output type: `hard` (bits).
    """

    code: abc.BlockCode

    def __post_init__(self) -> None:
        self._coset_leaders = self.code.coset_leaders()

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.SyndromeTableDecoder(code)
            >>> decoder.decode([[1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0]])
            array([[1, 1, 0, 0],
                   [1, 0, 1, 1]])
        """

        @blockwise(self.code.length)
        def decode(r: npt.NDArray[np.integer]):
            s = self.code.check(r)
            e_hat = self._coset_leaders[bits_to_int(s)]
            v_hat = np.bitwise_xor(r, e_hat)
            u_hat = self.code.project_word(v_hat)
            return u_hat

        return decode(input)
