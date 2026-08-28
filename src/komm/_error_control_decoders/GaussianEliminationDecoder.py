from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.decorators import blockwise, vectorize, with_pbar
from .._util.matrices import pseudo_inverse
from .util import get_pbar


@dataclass
class GaussianEliminationDecoder(abc.BlockDecoder[abc.BlockCode]):
    r"""
    Gaussian elimination decoder for general [block codes](/ref/BlockCode) over the [binary erasure channel](/ref/BinaryErasureChannel). Let $\mathcal{K}$ be the set of unerased positions of the received word $r$. This decoder returns a message $\hat{u}$ satisfying
    $$
        \hat{u} G_{\mathcal{K}} = r_{\mathcal{K}},
    $$
    where $G_{\mathcal{K}}$ is the submatrix of the generator matrix given by the columns in $\mathcal{K}$.

    Parameters:
        code: The block code to be used for decoding.

    Notes:
        - Input type: `erasure` (bits, with `2` denoting an erasure).
        - Output type: `hard` (bits).
        - The system is solved in $O(n^3)$ time, whereas maximum-likelihood decoding over the [binary symmetric channel](/ref/BinarySymmetricChannel) is NP-hard.
        - If the system has more than one solution, the decoder returns one of them. Since all of them are equally likely, the decoder is optimal regardless of the choice.
    """

    code: abc.BlockCode

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.HammingCode(3)
            >>> decoder = komm.GaussianEliminationDecoder(code)

            >>> decoder.decode([[1, 1, 0, 2, 0, 1, 2], [2, 2, 1, 1, 2, 1, 0]])
            array([[1, 1, 0, 0],
                   [1, 0, 1, 1]])

            >>> decoder.decode([2, 2, 2, 2, 2, 2, 2])  # Every position erased
            array([0, 0, 0, 0])
        """

        @blockwise(self.code.length)
        @vectorize
        @with_pbar(get_pbar(np.size(input) // self.code.length, "Gaussian elimination"))
        def decode(r: npt.NDArray[np.integer]):
            known = r != 2
            G_known = self.code.generator_matrix[:, known]
            u_hat = pseudo_inverse(G_known.T) @ r[known] % 2
            return u_hat

        return decode(input)
