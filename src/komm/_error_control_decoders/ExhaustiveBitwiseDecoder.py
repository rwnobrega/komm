from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.bit_operations import int_to_bits
from .._util.decorators import blockwise


@dataclass
class ExhaustiveBitwiseDecoder(abc.CodewordDecoder[abc.BlockCode]):
    r"""
    Exhaustive bitwise decoder for general [block codes](/ref/BlockCode). This decoder computes the a posteriori L-value of each bit by summing over all possible codewords, assuming equiprobable messages. Its hard decisions, the signs of the L-values, are the bitwise maximum a posteriori (MAP) decisions, and may not form a codeword. For more details, see <cite>MacK03, Sec. 25.1</cite>.

    Parameters:
        code: The block code to be used for decoding.
        output_type: The type of the output. Either `'hard'` or `'soft'`. Default is `'soft'`.

    Notes:
        - Input type: `soft` (L-values).
        - Output type: `hard` (bits) or `soft` (L-values).
    """

    code: abc.BlockCode
    output_type: Literal["hard", "soft"] = "soft"

    def __post_init__(self) -> None:
        if self.output_type not in ["hard", "soft"]:
            raise ValueError("'output_type' must be 'hard' or 'soft'")
        k = self.code.dimension
        self._codewords = self.code.codewords()
        self._messages = int_to_bits(range(2**k), width=k).reshape(-1, k)
        self._polar = (-1) ** self._codewords

    def _decode(
        self, input: npt.ArrayLike, bits: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.integer | np.floating]:
        @blockwise(self.code.length)
        def decode(li: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            metrics = 0.5 * li @ self._polar.T
            return _marginalize(metrics, bits)

        output = decode(input)
        if self.output_type == "hard":
            output = (output < 0.0).astype(int)
        return output

    def decode_to_codeword(
        self, input: npt.ArrayLike
    ) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)

            >>> decoder = komm.ExhaustiveBitwiseDecoder(code)
            >>> decoder.decode_to_codeword([-1.3, -0.8, +1.1, -0.8, +1.2, -0.2, -1.4])
            array([-1.4867878 , -0.47876436,  1.30351111, -0.2634431 ,  1.38090385,
                   -0.20271875, -1.57367212])

            >>> decoder = komm.ExhaustiveBitwiseDecoder(code, output_type="hard")
            >>> decoder.decode_to_codeword([-1.3, -0.8, +1.1, -0.8, +1.2, -0.2, -1.4])
            array([1, 1, 0, 1, 0, 1, 1])
        """
        return self._decode(input, self._codewords)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)

            >>> decoder = komm.ExhaustiveBitwiseDecoder(code)
            >>> decoder.decode([-1.3, -0.8, +1.1, -0.8, +1.2, -0.2, -1.4])
            array([-1.4867878 , -0.47876436,  1.30351111, -0.2634431 ])

            >>> decoder = komm.ExhaustiveBitwiseDecoder(code, output_type="hard")
            >>> decoder.decode([-1.3, -0.8, +1.1, -0.8, +1.2, -0.2, -1.4])
            array([1, 1, 0, 1])
        """
        return self._decode(input, self._messages)


def _marginalize(
    metrics: npt.NDArray[np.floating], bits: npt.NDArray[np.integer]
) -> npt.NDArray[np.floating]:
    # L-value of each bit, in log domain
    lo = [
        np.logaddexp.reduce(np.where(b == 0, metrics, -np.inf), axis=-1)
        - np.logaddexp.reduce(np.where(b == 1, metrics, -np.inf), axis=-1)
        for b in bits.T
    ]
    return np.stack(lo, axis=-1)
