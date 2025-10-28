from functools import cache

import numpy as np
import numpy.typing as npt

from .._util.docs import mkdocstrings
from ..types import Array1D
from .FixedToVariableCode import FixedToVariableCode
from .util import PrefixFreePMFCode


@mkdocstrings(filters=["!.*"])
class ShannonCode(PrefixFreePMFCode, FixedToVariableCode):
    r"""
    Binary Shannon code. For a given pmf $p$ over $\mathcal{X}$, it is a [fixed-to-variable length code](/ref/FixedToVariableCode) in which the length of the codeword $\Enc(\mathbf{x})$ associated with a source word $\mathbf{x} \in \mathcal{X}^k$ is given by
    $$
        \ell(\mathbf{x}) = \left\lceil \log_2 \frac{1}{p(\mathbf{x})} \right\rceil.
    $$
    This function implements the lexicographic order assignment as described in [Wikipedia: Shannon–Fano coding](https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano_coding).

    Notes:
        Shannon codes are always [prefix-free](/ref/FixedToVariableCode/#is_prefix_free) (hence [uniquely decodable](/ref/FixedToVariableCode/#is_uniquely_decodable)).

    Parameters:
        pmf: The pmf $p$ to be considered. It must be a one-dimensional array of floats of size $|\mathcal{X}|$. The elements must be non-negative and sum to $1$.

        source_block_size: The source block size $k$. The default value is $k = 1$.

    Examples:
        >>> pmf = [0.8, 0.1, 0.1]

        >>> code = komm.ShannonCode(pmf)
        >>> code.enc_mapping
        {(0,): (0,),
         (1,): (1, 0, 0, 0),
         (2,): (1, 0, 0, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(1.6)

        >>> code = komm.ShannonCode(pmf, 2)
        >>> code.enc_mapping
        {(0, 0): (0,),
         (0, 1): (1, 0, 0, 0),
         (0, 2): (1, 0, 0, 1),
         (1, 0): (1, 0, 1, 0),
         (1, 1): (1, 1, 0, 0, 0, 0, 0),
         (1, 2): (1, 1, 0, 0, 0, 0, 1),
         (2, 0): (1, 0, 1, 1),
         (2, 1): (1, 1, 0, 0, 0, 1, 0),
         (2, 2): (1, 1, 0, 0, 0, 1, 1)}
        >>> code.rate(pmf)  # doctest: +FLOAT_CMP
        np.float64(1.1)
    """

    def __init__(self, pmf: npt.ArrayLike, source_block_size: int = 1):
        super().__init__(pmf, source_block_size, shannon_code_lengths)


def shannon_code_lengths(pmf: Array1D[np.floating]) -> Array1D[np.integer]:
    lengths = np.zeros_like(pmf, dtype=int)
    mask = pmf > 0
    if np.sum(pmf**2) == 1:  # Deterministic case
        lengths[mask] = 1
    else:
        lengths[mask] = np.ceil(np.log2(1 / pmf[mask])).astype(int)
    return lengths
