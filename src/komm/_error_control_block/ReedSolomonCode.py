from functools import partial, reduce

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

from .._algebra import bifield
from .._algebra.FiniteBifield import FiniteBifield
from .._util.docs import mkdocstrings
from .SystematicBlockCode import SystematicBlockCode


@mkdocstrings(filters=["!.*"])
@typechecked
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
        - For $\mu = 8$ and $\delta = 33$ it is a $(255, 223)$ Reed–Solomon code, which corrects $16$ symbol errors. Its binary image is a $(2040, 1784)$ code.

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

        >>> code = komm.ReedSolomonCode(mu=8, delta=33)
        >>> (code.length, code.dimension, code.redundancy)
        (2040, 1784, 256)
    """

    def __init__(self, mu: int, delta: int) -> None:
        if not mu >= 2:
            raise ValueError("'mu' must satisfy mu >= 2")
        if not 2 <= delta <= 2**mu - 1:
            raise ValueError("'delta' must satisfy 2 <= delta <= 2**mu - 1")
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
