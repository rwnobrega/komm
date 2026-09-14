from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_block.SingleParityCheckCode import SingleParityCheckCode
from .._util.decorators import blockwise, vectorize


@dataclass
class WagnerDecoder(abc.CodewordDecoder[SingleParityCheckCode]):
    r"""
    Wagner decoder for [single parity-check codes](/ref/SingleParityCheckCode). For more details, see <cite>CF07, Sec. III.C</cite>.

    Parameters:
        code: The single parity-check code to be used for decoding.

    Notes:
        - Input type: `soft` (L-values).
        - Output type: `hard` (bits).
    """

    code: SingleParityCheckCode

    def decode_to_codeword(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.SingleParityCheckCode(4)
            >>> decoder = komm.WagnerDecoder(code)
            >>> decoder.decode_to_codeword([
            ...     [1.52, -0.36, 1.56, 0.82],
            ...     [-0.75,  1.20, -2.11, 1.73],
            ... ])
            array([[0, 0, 0, 0],
                   [1, 0, 1, 0]])
        """

        @blockwise(self.code.length)
        @vectorize
        def decode_to_codeword(r: npt.NDArray[np.floating]):
            v_hat = (r < 0).astype(int)
            if np.count_nonzero(v_hat) % 2 != 0:
                i = np.argmin(np.abs(r))
                v_hat[i] ^= 1
            return v_hat

        return decode_to_codeword(input)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.SingleParityCheckCode(4)
            >>> decoder = komm.WagnerDecoder(code)
            >>> decoder.decode([
            ...     [1.52, -0.36, 1.56, 0.82],
            ...     [-0.75,  1.20, -2.11, 1.73],
            ... ])
            array([[0, 0, 0],
                   [1, 0, 1]])
        """
        return self.code.project_word(self.decode_to_codeword(input))
