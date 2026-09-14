from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.decorators import blockwise, vectorize, with_pbar
from .._util.validators import validate_integer_range
from .util import get_pbar


@dataclass
class PeelingDecoder(abc.CodewordDecoder[abc.BlockCode]):
    r"""
    Peeling decoder for general [block codes](/ref/BlockCode) over the [binary erasure channel](/ref/BinaryErasureChannel). This decoder resolves erased positions one at a time, from parity checks with a single erased position, and stops when none is left. For more details, see <cite>RU08, Sec. 3.19</cite>.

    Parameters:
        code: The block code to be used for decoding.

    Notes:
        - Input type: `erasure` (bits, with `2` denoting an erasure).
        - Output type: `erasure` (bits, with `2` denoting an undetermined position).
    """

    code: abc.BlockCode

    def __post_init__(self) -> None:
        n, m, H = self.code.length, self.code.redundancy, self.code.check_matrix
        self._chks_of = [frozenset(np.flatnonzero(H[:, j]).tolist()) for j in range(n)]
        self._vars_of = [frozenset(np.flatnonzero(H[i, :]).tolist()) for i in range(m)]

    def decode_to_codeword(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.PeelingDecoder(code)
            >>> decoder.decode_to_codeword([2, 1, 0, 2, 2, 1, 1])
            array([1, 1, 0, 0, 0, 1, 1])
            >>> decoder.decode_to_codeword([2, 2, 0, 2, 0, 1, 1])  # Stopping set: peeling stalls
            array([2, 2, 0, 2, 0, 1, 1])
            >>> decoder.decode_to_codeword([1, 0, 2, 1, 2, 2, 2])
            array([1, 0, 2, 1, 0, 2, 2])
        """
        input = validate_integer_range(input, low=0, high=3)

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "peeling"))
        def decode_to_codeword(r: npt.NDArray[np.integer]):
            # See [RU08, Example 3.104, p. 118].
            H = self.code.check_matrix
            v_hat = r.copy()
            known = v_hat != 2
            acc = (H @ np.where(known, v_hat, 0)) % 2
            erased = set(np.flatnonzero(~known).tolist())
            residual = [erased & s for s in self._vars_of]
            stack = [i for i, s in enumerate(residual) if len(s) == 1]
            while stack:
                i = stack.pop()
                if len(residual[i]) != 1:
                    continue
                j = residual[i].pop()
                v_hat[j] = acc[i]
                for i0 in self._chks_of[j] - {i}:
                    acc[i0] ^= v_hat[j]
                    residual[i0].discard(j)
                    if len(residual[i0]) == 1:
                        stack.append(i0)
            return v_hat

        return decode_to_codeword(input)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Note:
            For this decoder, the message is read off from the resolved positions through a right inverse of the generator matrix; message bits that depend on an unresolved position are marked as erasures.

        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.PeelingDecoder(code)
            >>> decoder.decode([2, 1, 0, 2, 2, 1, 1])
            array([1, 1, 0, 0])
            >>> decoder.decode([2, 2, 0, 2, 0, 1, 1])  # Stopping set: peeling stalls
            array([2, 2, 0, 2])
            >>> decoder.decode([1, 0, 2, 1, 2, 2, 2])
            array([1, 0, 2, 1])
        """
        return self.code.project_word_with_erasures(self.decode_to_codeword(input))
