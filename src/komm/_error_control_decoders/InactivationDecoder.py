from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.decorators import blockwise, vectorize, with_pbar
from .._util.matrices import left_null_matrix, pseudo_inverse
from .._util.validators import validate_integer_range
from .util import get_pbar, peel


@dataclass
class InactivationDecoder(abc.BlockDecoder[abc.BlockCode]):
    r"""
    Inactivation decoder for general [block codes](/ref/BlockCode) over the [binary erasure channel](/ref/BinaryErasureChannel). It runs the [peeling decoder](/ref/PeelingDecoder) and, instead of giving up when it stalls, handles the surviving unknowns (called *inactivated*) by Gaussian elimination.

    Parameters:
        code: The block code to be used for decoding.

    Notes:
        - Input type: `erasure` (bits, with `2` denoting an erasure).
        - Output type: `erasure` (bits, with `2` denoting an undetermined position).
        - The output is identical to that of the [Gaussian elimination decoder](/ref/GaussianEliminationDecoder), but only the inactivated unknowns reach the elimination step. For sparse check matrices, they are few, and decoding is much faster.
        - This is the algorithm used to decode RaptorQ fountain codes.
    """

    code: abc.BlockCode

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Raises:
            ValueError: If the input contains entries outside of $\\{ 0, 1, 2 \\}$.

        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.InactivationDecoder(code)
            >>> decoder.decode([2, 2, 0, 2, 0, 1, 1])  # Stopping set
            array([1, 1, 0, 0])
            >>> decoder.decode([2, 0, 1, 1, 2, 2, 0])
            array([2, 0, 1, 1])
            >>> decoder.decode([2, 2, 2, 2, 2, 2, 2])
            array([2, 2, 2, 2])
        """
        input = validate_integer_range(input, low=0, high=3)

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "inactivation"))
        def decode(r: npt.NDArray[np.integer]):
            H = self.code.check_matrix
            erased = r == 2
            A = H[:, erased].astype(bool)
            b = H[:, ~erased] @ r[~erased] % 2
            x, inactive, b = peel(A, b)
            v_hat = r.copy()
            if inactive.any():
                x[inactive] = pseudo_inverse(A[:, inactive]) @ b % 2
            v_hat[erased] = x
            u_hat = self.code.project_word(v_hat)
            if inactive.any():
                # Codewords supported inside the erasures.
                basis = left_null_matrix(A[:, inactive].T)
                if basis.size:
                    words = np.zeros((basis.shape[0], self.code.length), dtype=int)
                    words[:, np.flatnonzero(erased)[inactive]] = basis
                    free = self.code.project_word(words).any(axis=0).astype(bool)
                    u_hat[free] = 2
            return u_hat

        return decode(input)
