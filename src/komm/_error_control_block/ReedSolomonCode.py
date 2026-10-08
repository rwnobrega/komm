from functools import partial, reduce

import numpy as np
import numpy.typing as npt

from .._algebra import bifield
from .._algebra.FiniteBifield import FiniteBifield
from .._util.docs import mkdocstrings
from .._util.validators import validate_integer
from .SystematicBlockCode import SystematicBlockCode


@mkdocstrings(filters=["!.*"])
class ReedSolomonCode(SystematicBlockCode):
    r"""
    Reed–Solomon code. For given parameters $\mu \geq 2$ and $\delta$ satisfying $2 \leq \delta \leq 2^{\mu} - 1$, a *Reed–Solomon code* is a cyclic code over $\mathrm{GF}(2^\mu)$ with generator polynomial given by
    $$
        g(X) = (X + \alpha) (X + \alpha^2) \cdots (X + \alpha^{\delta - 1}),
    $$
    where $\alpha$ is a primitive element of $\mathrm{GF}(2^\mu)$. The resulting code has the following parameters, in symbols:

    - Length: $n = 2^{\mu} - 1$
    - Dimension: $k = 2^{\mu} - \delta$
    - Redundancy: $m = \delta - 1$
    - Minimum distance: $d = \delta$

    This class represents the *binary image* of the code, a [linear block code](/ref/BlockCode) in which each symbol is replaced by the $\mu$ coefficients of its polynomial representation, in increasing order of degree. The binary image has the following parameters, in bits:

    - Length: $\mu n$
    - Dimension: $\mu k$
    - Redundancy: $\mu m$
    - Minimum distance: $d \geq \delta$

    Only *narrow-sense* and *primitive* Reed–Solomon codes are implemented. For more details, see <cite>LC04, Sec. 7.3</cite>.

    Notes:
        - Reed–Solomon codes are maximum distance separable (MDS) codes, that is, $d = n - k + 1$, in symbols.

    Parameters:
        mu: The parameter $\mu$ of the code. Must satisfy $\mu \geq 2$.
        delta: The minimum distance $\delta$ of the code, in symbols. Must satisfy $2 \leq \delta \leq 2^{\mu} - 1$.

    This class represents the code in [systematic form](/ref/SystematicBlockCode), with the information set on the right.

    Examples:
        >>> code = komm.ReedSolomonCode(mu=3, delta=5)
        >>> (code.length, code.dimension, code.redundancy)
        (21, 9, 12)
        >>> code.minimum_distance()
        6

        >>> code = komm.ReedSolomonCode(mu=8, delta=33)  # (255, 223) in symbols
        >>> (code.length, code.dimension, code.redundancy)
        (2040, 1784, 256)
    """

    def __init__(self, mu: int, delta: int) -> None:
        mu = validate_integer(mu, low=2)
        rule = "2 <= delta <= 2**mu - 1"
        delta = validate_integer(delta, low=2, high=2**mu, rule=rule)
        self.mu = mu
        self.delta = delta
        self.field = FiniteBifield(mu)
        self.alpha = self.field.primitive_element
        super().__init__(
            parity_submatrix=reed_solomon_parity_submatrix(self.field, delta),
            information_set="right",
        )

    def __repr__(self) -> str:
        args = f"mu={self.mu}, delta={self.delta}"
        return f"{self.__class__.__name__}({args})"


def reed_solomon_parity_submatrix(
    field: FiniteBifield,
    delta: int,
) -> npt.NDArray[np.integer]:
    n, m = field.order - 1, delta - 1
    # g(X) = (X + α)·(X + α^2)· ⋯ ·(X + α^m)
    roots = bifield.power(field, int(field.primitive_element), np.arange(1, delta))
    factors = np.column_stack([roots, np.ones_like(roots)])  # [α^i 1] <-> (X + α^i)
    g = reduce(partial(bifield.convolve, field), factors)
    # Parity symbols of X^m, ..., X^(n - 1).
    _, parities = bifield.deconvolve(field, np.eye(n, dtype=int)[m:], g)
    return bifield.binary_matrix(field, parities)
