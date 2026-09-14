from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.bit_operations import bits_to_int


@dataclass
class SyndromeTableDecoder(abc.CodewordDecoder[abc.BlockCode]):
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

    def decode_to_codeword(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.SyndromeTableDecoder(code)
            >>> decoder.decode_to_codeword([
            ...     [1, 1, 0, 1, 0, 1, 1],
            ...     [1, 0, 1, 1, 0, 0, 0],
            ... ])
            array([[1, 1, 0, 0, 0, 1, 1],
                   [1, 0, 1, 1, 0, 1, 0]])
        """
        r = np.asarray(input)
        s = self.code.check(r)
        e_hat = self._coset_leaders[bits_to_int(s, width=self.code.redundancy)]
        v_hat = np.bitwise_xor(r, e_hat.reshape(r.shape))
        return v_hat

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.SyndromeTableDecoder(code)
            >>> decoder.decode([
            ...     [1, 1, 0, 1, 0, 1, 1],
            ...     [1, 0, 1, 1, 0, 0, 0],
            ... ])
            array([[1, 1, 0, 0],
                   [1, 0, 1, 1]])
        """
        return self.code.project_word(self.decode_to_codeword(input))
