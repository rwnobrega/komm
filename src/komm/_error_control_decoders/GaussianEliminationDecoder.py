from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.decorators import blockwise, vectorize, with_pbar
from .._util.matrices import left_null_matrix, pseudo_inverse
from .._util.validators import validate_integer_range
from .util import get_pbar


@dataclass
class GaussianEliminationDecoder(abc.BlockDecoder[abc.BlockCode]):
    r"""
    Gaussian elimination decoder for general [block codes](/ref/BlockCode) over the [binary erasure channel](/ref/BinaryErasureChannel). Let $\mathcal{E}$ be the set of erased positions of the received word $r$. The compatible messages $\hat{u}$ are the solutions of the linear system
    $$
        \hat{u} G_{\bar{\mathcal{E}}} = r_{\bar{\mathcal{E}}},
    $$
    where $G_{\bar{\mathcal{E}}}$ is the submatrix of the generator matrix given by the columns not in $\mathcal{E}$. This decoder returns the message bits shared by all solutions; the remaining ones are marked as erasures.

    Parameters:
        code: The block code to be used for decoding.

    Notes:
        - Input type: `erasure` (bits, with `2` denoting an erasure).
        - Output type: `erasure` (bits, with `2` denoting an undetermined position).
    """

    code: abc.BlockCode

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Raises:
            ValueError: If the input contains entries outside of $\\{ 0, 1, 2 \\}$.

        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.GaussianEliminationDecoder(code)
            >>> decoder.decode([2, 1, 0, 2, 2, 1, 1])
            array([1, 1, 0, 0])
            >>> decoder.decode([2, 2, 0, 2, 0, 1, 1])  # Stopping set, but still recoverable
            array([1, 1, 0, 0])
            >>> decoder.decode([2, 0, 1, 1, 2, 2, 0])  # Erased codeword support: unrecoverable
            array([2, 0, 1, 1])
            >>> decoder.decode([2, 2, 2, 2, 2, 2, 2])
            array([2, 2, 2, 2])
        """
        input = validate_integer_range(input, low=0, high=3)

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "Gaussian elimination"))
        def decode(r: npt.NDArray[np.integer]):
            known = r != 2
            G_known = self.code.generator_matrix[:, known]
            u_hat = r[known] @ pseudo_inverse(G_known) % 2
            free = left_null_matrix(G_known).any(axis=0).astype(bool)
            u_hat[free] = 2
            return u_hat

        return decode(input)
