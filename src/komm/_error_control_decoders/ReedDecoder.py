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

    # `__call__`

    :::komm.abc.BlockDecoder.BlockDecoder.__call__

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

    code: ReedMullerCode
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        self._reed_partitions = self.code.reed_partitions()

    @vectorized_method
    def _decode_hard(self, r: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        u_hat = np.empty(self.code.dimension, dtype=int)
        bx = r.copy()
        for i, partition in enumerate(self._reed_partitions):
            checksums = np.count_nonzero(bx[partition], axis=1) % 2
            u_hat[i] = np.count_nonzero(checksums) > len(checksums) // 2
            bx ^= u_hat[i] * self.code.generator_matrix[i]
        return u_hat

    @vectorized_method
    def _decode_soft(self, r: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        u_hat = np.empty(self.code.dimension, dtype=int)
        bx = (r < 0).astype(int)
        for i, partition in enumerate(self._reed_partitions):
            checksums = np.count_nonzero(bx[partition], axis=1) % 2
            min_reliability = np.min(np.abs(r[partition]), axis=1)
            decision_var = (1 - 2 * checksums) @ min_reliability
            u_hat[i] = decision_var < 0
            bx ^= u_hat[i] * self.code.generator_matrix[i]
        return u_hat

    def _decode(
        self, r: npt.NDArray[np.integer | np.floating]
    ) -> npt.NDArray[np.integer]:
        if self.input_type == "hard":
            return self._decode_hard(r)
        else:  # self.input_type == "soft"
            return self._decode_soft(r)
