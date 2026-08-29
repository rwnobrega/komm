from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.decorators import blockwise, vectorize, with_pbar
from .._util.validators import validate_integer_range
from .util import get_pbar, peel


@dataclass
class PeelingDecoder(abc.BlockDecoder[abc.BlockCode]):
    r"""
    Peeling decoder for general [block codes](/ref/BlockCode) over the [binary erasure channel](/ref/BinaryErasureChannel). Let $\mathcal{E}$ be the set of erased positions of the received word $r$. The erased bits $x$ satisfy the linear system
    $$
        H_{\mathcal{E}} x^\transpose = H_{\bar{\mathcal{E}}} r_{\bar{\mathcal{E}}}^\transpose,
    $$
    where $H_{\mathcal{E}}$ is the submatrix of the check matrix given by the columns in $\mathcal{E}$. Rows with a single unknown are solved and substituted, one at a time, until no such row is left. This is belief propagation specialized to the erasure channel.

    Parameters:
        code: The block code to be used for decoding.

    Notes:
        - Input type: `erasure` (bits, with `2` denoting an erasure).
        - Output type: `erasure` (bits, with `2` denoting an undetermined position).
        - Decoding fails when the erased positions contain a *stopping set*, that is, a set of columns such that every row of $H$ meets it in zero or at least two positions. In that case, every position of the output is erased. The decoder is therefore suboptimal, unlike the [Gaussian elimination decoder](/ref/GaussianEliminationDecoder).
        - Performance depends on the check matrix, not only on the code. The check matrix derived from the generator matrix is usually dense, which is the worst case here. To get the intended behavior, build the code from a sparse check matrix, as in `komm.BlockCode(check_matrix=H)`.
    """

    code: abc.BlockCode

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Raises:
            ValueError: If the input contains entries outside of $\\{ 0, 1, 2 \\}$.

        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.PeelingDecoder(code)
            >>> decoder.decode([1, 1, 0, 2, 0, 1, 2])
            array([1, 1, 0, 0])
            >>> decoder.decode([2, 2, 0, 2, 0, 1, 1])  # Stopping set
            array([2, 2, 2, 2])
            >>> decoder.decode([2, 0, 1, 1, 2, 2, 0])  # Erased codeword support
            array([2, 2, 2, 2])
            >>> decoder.decode([2, 2, 2, 2, 2, 2, 2])
            array([2, 2, 2, 2])
        """
        input = validate_integer_range(input, low=0, high=3)

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "peeling"))
        def decode(r: npt.NDArray[np.integer]):
            H = self.code.check_matrix
            erased = r == 2
            A = H[:, erased].astype(bool)
            b = H[:, ~erased] @ r[~erased] % 2
            x, unknown, _ = peel(A, b)
            if unknown.any():
                return np.full(self.code.dimension, 2)
            v_hat = r.copy()
            v_hat[erased] = x
            u_hat = self.code.project_word(v_hat)
            return u_hat

        return decode(input)
