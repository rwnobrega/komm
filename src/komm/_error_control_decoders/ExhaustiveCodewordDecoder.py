from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.decorators import blockwise, chunkwise, with_pbar
from .util import get_pbar


@dataclass
class ExhaustiveCodewordDecoder(abc.CodewordDecoder[abc.BlockCode]):
    r"""
    Exhaustive codeword decoder for general [block codes](/ref/BlockCode). This decoder implements a brute-force search over all possible codewords to find the one that is closest (in terms of Hamming distance, for hard-decision decoding, or Euclidean distance, for soft-decision decoding) to the received word. This is the maximum-likelihood (ML) codeword, which is also the maximum a posteriori (MAP) codeword when messages are equiprobable. For more details, see <cite>MacK03, Sec. 25.1</cite>.

    Parameters:
        code: The block code to be used for decoding.
        input_type: The type of the input. Either `'hard'` or `'soft'`. Default is `'hard'`.

    Notes:
        - Input type: `hard` (bits) or `soft` (L-values).
        - Output type: `hard` (bits).
    """

    code: abc.BlockCode
    input_type: Literal["hard", "soft"] = "hard"

    def __post_init__(self) -> None:
        k = self.code.dimension
        self._codewords = self.code.codewords()
        # About 64 MiB of metrics
        self._chunk_size = max(1, 2**26 // (8 * 2**k * self.code.length))

    def decode_to_codeword(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)

            >>> decoder = komm.ExhaustiveCodewordDecoder(code, input_type="hard")
            >>> decoder.decode_to_codeword([
            ...     [1, 1, 0, 1, 0, 1, 1],
            ...     [1, 0, 1, 1, 0, 0, 0],
            ... ])
            array([[1, 1, 0, 0, 0, 1, 1],
                   [1, 0, 1, 1, 0, 1, 0]])

            >>> decoder = komm.ExhaustiveCodewordDecoder(code, input_type="soft")
            >>> decoder.decode_to_codeword([-1.3, -0.8, +1.1, -0.8, +1.2, -0.2, -1.4])
            array([1, 1, 0, 0, 0, 1, 1])
        """

        @blockwise(self.code.length)
        @chunkwise(self._chunk_size)
        @with_pbar(get_pbar(np.size(input) // self.code.length, "exhaustive codeword"))
        def decode_to_codeword(r: npt.NDArray[np.integer]):
            if self.input_type == "hard":
                ds = r[..., np.newaxis, :] != self._codewords
            else:
                ds = -r[..., np.newaxis, :] * (-1) ** self._codewords
            metrics = np.sum(ds, axis=-1)
            v_hat = self._codewords[np.argmin(metrics, axis=-1)]
            return v_hat

        return decode_to_codeword(input)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)

            >>> decoder = komm.ExhaustiveCodewordDecoder(code, input_type="hard")
            >>> decoder.decode([
            ...     [1, 1, 0, 1, 0, 1, 1],
            ...     [1, 0, 1, 1, 0, 0, 0],
            ... ])
            array([[1, 1, 0, 0],
                   [1, 0, 1, 1]])

            >>> decoder = komm.ExhaustiveCodewordDecoder(code, input_type="soft")
            >>> decoder.decode([-1.3, -0.8, +1.1, -0.8, +1.2, -0.2, -1.4])
            array([1, 1, 0, 0])
        """
        return self.code.project_word(self.decode_to_codeword(input))
