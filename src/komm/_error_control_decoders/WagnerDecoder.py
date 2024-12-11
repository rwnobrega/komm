from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_block import SingleParityCheckCode
from .._util.decorators import vectorized_method


@dataclass
class WagnerDecoder(abc.BlockDecoder[SingleParityCheckCode]):
    r"""
    Wagner decoder for [single parity-check codes](/ref/SingleParityCheckCode). For more information, see <cite>CF07, Sec. III.C</cite>.

    Parameters:
        code: The single parity-check code to be used for decoding.

    Parameters: Input:
        r: The input received word(s). Can be a single received word of length $n$ or a multidimensional array where the last dimension has length $n$.

    Parameters: Output:
        u_hat: The output message(s). Has the same shape as the input, with the last dimension reduced from $n$ to $k$.

    Notes:
        - Input type: `soft`.
        - Output type: `hard`.

    Examples:
        >>> code = komm.SingleParityCheckCode(4)
        >>> decoder = komm.WagnerDecoder(code)
        >>> decoder([[1.52, -0.36, 1.56, 0.82], [-0.75,  1.20 , -2.11,  1.73]])
        array([[0, 0, 0],
               [1, 0, 1]])
    """

    code: SingleParityCheckCode

    @vectorized_method
    def _decode(self, r: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        v_hat = (r < 0).astype(int)
        if np.count_nonzero(v_hat) % 2 != 0:
            i = np.argmin(np.abs(r))
            v_hat[i] ^= 1
        u_hat = self.code.inverse_encode(v_hat)
        return u_hat
