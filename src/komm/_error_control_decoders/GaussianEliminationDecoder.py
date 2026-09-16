from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.decorators import blockwise, vectorize, with_pbar
from .._util.matrices import solution_set
from .._util.validators import validate_integer_range
from .util import get_pbar


@dataclass
class GaussianEliminationDecoder(abc.CodewordDecoder[abc.BlockCode]):
    r"""
    Gaussian elimination decoder for general [block codes](/ref/BlockCode) over the [binary erasure channel](/ref/BinaryErasureChannel). This decoder performs bit-wise MAP decoding: it solves the linear system relating the erased positions to the received ones, and returns the bits shared by all its solutions. For more details, see <cite>RU08, Sec. 3.2</cite>.

    Parameters:
        code: The block code to be used for decoding.

    Notes:
        - Input type: `erasure` (bits, with `2` denoting an erasure).
        - Output type: `erasure` (bits, with `2` denoting an undetermined position).
    """

    code: abc.BlockCode

    def decode_to_codeword(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.GaussianEliminationDecoder(code)
            >>> decoder.decode_to_codeword([2, 1, 0, 2, 2, 1, 1])
            array([1, 1, 0, 0, 0, 1, 1])
            >>> decoder.decode_to_codeword([2, 2, 0, 2, 0, 1, 1])  # Stopping set, but still recoverable
            array([1, 1, 0, 0, 0, 1, 1])
            >>> decoder.decode_to_codeword([1, 0, 2, 1, 2, 2, 2])
            array([1, 0, 2, 1, 0, 2, 2])
        """
        input = validate_integer_range(input, low=0, high=3)
        G, H = self.code.generator_matrix, self.code.check_matrix

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "Gaussian elimination"))
        def decode_to_codeword(r: npt.NDArray[np.integer]):
            # Solve for the erased positions or the message bits, whichever are fewer.
            if np.count_nonzero(r == 2) < self.code.dimension:
                v_hat, W = _compatible_codewords(H, r)
                v_hat[W.any(axis=0)] = 2
            else:
                u0, N = _compatible_messages(G, r)
                v_hat = u0 @ G % 2
                v_hat[(N @ G % 2).any(axis=0)] = 2
            return v_hat

        return decode_to_codeword(input)

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.GaussianEliminationDecoder(code)
            >>> decoder.decode([2, 1, 0, 2, 2, 1, 1])
            array([1, 1, 0, 0])
            >>> decoder.decode([2, 2, 0, 2, 0, 1, 1])  # Stopping set, but still recoverable
            array([1, 1, 0, 0])
            >>> decoder.decode([1, 0, 2, 1, 2, 2, 2])
            array([1, 0, 2, 1])
        """
        input = validate_integer_range(input, low=0, high=3)
        G, H = self.code.generator_matrix, self.code.check_matrix
        G_r_inv = self.code.generator_matrix_right_inverse

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "Gaussian elimination"))
        def decode(r: npt.NDArray[np.integer]):
            # Solve for the erased positions or the message bits, whichever are fewer.
            if np.count_nonzero(r == 2) < self.code.dimension:
                v0, W = _compatible_codewords(H, r)
                u_hat = v0 @ G_r_inv % 2
                u_hat[(W @ G_r_inv % 2).any(axis=0)] = 2
            else:
                u_hat, N = _compatible_messages(G, r)
                u_hat[N.any(axis=0)] = 2
            return u_hat

        return decode(input)


def _compatible_codewords(
    H: npt.NDArray[np.integer], r: npt.NDArray[np.integer]
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    # Returns (v0, W) such that the codewords compatible with r are v0 + span(W).
    # See [RU08, Sec. 3.2, pp. 72–74].
    erased = r == 2
    s = H @ np.where(erased, 0, r) % 2
    v0_erased, W_erased = solution_set(H[:, erased].T, s)
    # The unknowns live in E; expand them back to length n.
    v0 = r.copy()
    v0[erased] = v0_erased
    W = np.zeros((W_erased.shape[0], r.size), dtype=int)
    W[:, erased] = W_erased
    return v0, W


def _compatible_messages(
    G: npt.NDArray[np.integer], r: npt.NDArray[np.integer]
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    # Returns (u0, N) such that the messages compatible with r are u0 + span(N).
    known = r != 2
    return solution_set(G[:, known], r[known])
