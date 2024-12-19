from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._error_control_block import BlockCode


@dataclass
class ExhaustiveSearchDecoder(abc.BlockDecoder[BlockCode]):
    r"""
    Exhaustive search decoder for general [block codes](/ref/BlockCode). This decoder implements a brute-force search over all possible codewords to find the one that is closest (in terms of Hamming distance, for hard-decision decoding, or Euclidean distance, for soft-decision decoding) to the received word.

    Parameters:
        code: The block code to be used for decoding.
        input_type: The type of the input. Either `'hard'` or `'soft'`. Default is `'hard'`.

    Notes:
        - Input type: `hard` or `soft`.
        - Output type: `hard`.

    # `__call__`

    :::komm.abc.BlockDecoder.BlockDecoder.__call__

    Examples:
        >>> code = komm.HammingCode(3)
        >>> decoder = komm.ExhaustiveSearchDecoder(code, input_type="hard")
        >>> decoder([[1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0]])
        array([[1, 1, 0, 0],
               [1, 0, 1, 1]])

        >>> code = komm.HammingCode(3)
        >>> decoder = komm.ExhaustiveSearchDecoder(code, input_type="soft")
        >>> decoder([[-1, -1, +1, -1, +1, -1, -1], [-1, +1, -1, -1, +1, +1, +1]])
        array([[1, 1, 0, 0],
               [1, 0, 1, 1]])
    """

    code: BlockCode
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        self._codewords = self.code.codewords()

    def _decode(
        self, r: npt.NDArray[np.float64 | np.integer]
    ) -> npt.NDArray[np.integer]:
        if self.input_type == "hard":
            ds = r[..., np.newaxis, :] != self._codewords
        else:
            ds = -r[..., np.newaxis, :] * (-1) ** self._codewords
        metrics = np.sum(ds, axis=-1)
        v_hat = self._codewords[np.argmin(metrics, axis=-1)]
        u_hat = self.code.inverse_encode(v_hat)
        return u_hat
