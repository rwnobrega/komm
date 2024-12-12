from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_block import ReedMullerCode
from .._util.decorators import vectorized_method


@dataclass
class ReedDecoder(abc.BlockDecoder[ReedMullerCode]):
    r"""
    Reed decoder for [Reed-Muller codes](/ref/ReedMullerCode). It's a majority-logic decoding algorithm. For more details, see [LC04, Sec 4.3 and 10.9.1] for hard-decision decoding, and [LC04, Sec 10.9.2] for soft-decision decoding.

    Parameters:
        code: The Reed-Muller code to be used for decoding.
        input_type: The type of the input. Either `'hard'` or `'soft'`. Default is `'hard'`.

    Notes:
        - Input type: `hard` or `soft`.
        - Output type: `hard`.

    :::komm.ReedDecoder.ReedDecoder._decode
    """

    code: ReedMullerCode
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        self.reed_partitions = self.code.reed_partitions()

    @vectorized_method
    def _decode_hard(self, input: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        output = np.empty(self.code.dimension, dtype=int)
        bx = input.copy()
        for i, partition in enumerate(self.reed_partitions):
            checksums = np.count_nonzero(bx[partition], axis=1) % 2
            output[i] = np.count_nonzero(checksums) > len(checksums) // 2
            bx ^= output[i] * self.code.generator_matrix[i]
        return output

    @vectorized_method
    def _decode_soft(self, input: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        output = np.empty(self.code.dimension, dtype=int)
        bx = (input < 0).astype(int)
        for i, partition in enumerate(self.reed_partitions):
            checksums = np.count_nonzero(bx[partition], axis=1) % 2
            min_reliability = np.min(np.abs(input[partition]), axis=1)
            decision_var = (1 - 2 * checksums) @ min_reliability
            output[i] = decision_var < 0
            bx ^= output[i] * self.code.generator_matrix[i]
        return output

    def _decode(
        self, input: npt.NDArray[np.integer | np.floating]
    ) -> npt.NDArray[np.integer]:
        r"""
        Parameters: Input:
            input: The input received word(s). Can be a single received word of length $n$ or a multidimensional array where the last dimension has length $n$.

        Returns: Output:
            output: The output message(s). Has the same shape as the input, with the last dimension reduced from $n$ to $k$.

        Examples:
            >>> code = komm.ReedMullerCode(1, 3)
            >>> decoder = komm.ReedDecoder(code, input_type="hard")
            >>> decoder([[0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 1, 1, 1]])
            array([[0, 0, 0, 0],
                   [0, 0, 0, 1]])

            >>> code = komm.ReedMullerCode(1, 3)
            >>> decoder = komm.ReedDecoder(code, input_type="soft")
            >>> decoder([+1.3, +1.0, +0.9, +0.4, -0.8, +0.2, +0.3, +0.8])
            array([0, 0, 0, 0])
        """
        if self.input_type == "hard":
            return self._decode_hard(input)
        else:  # self.input_type == "soft"
            return self._decode_soft(input)
