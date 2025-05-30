from functools import cached_property

import numpy as np
import numpy.typing as npt

from .._util import global_rng
from .._util.information_theory import PMF, LogBase, entropy


class DiscreteMemorylessSource:
    r"""
    Discrete memoryless source (DMS). It is defined by an *alphabet* $\mathcal{X}$ and a *probability mass function* (pmf) $p_X$. Here, for simplicity, the alphabet is always taken as $\mathcal{X} = [0 : |\mathcal{X}|)$. The pmf $p_X$ gives the probability of the source emitting the symbol $X = x$.

    Parameters:
        pmf: Either the source pmf $p_X$, in which case the element at position $x \in \mathcal{X}$ must be equal to $p_X(x)$, or an integer $n \geq 1$, in which case a uniform distribution over $n$ symbols is used.

    Examples:
        >>> komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
        DiscreteMemorylessSource(pmf=[0.5, 0.25, 0.125, 0.125])

        >>> komm.DiscreteMemorylessSource(4)
        DiscreteMemorylessSource(pmf=[0.25, 0.25, 0.25, 0.25])
    """

    def __init__(
        self,
        pmf: npt.ArrayLike | int,
        rng: np.random.Generator | None = None,
    ):
        if isinstance(pmf, int):
            if not pmf >= 1:
                raise ValueError("cardinality must be at least 1")
            pmf = np.full(pmf, 1 / pmf)
        self.pmf = PMF(pmf)
        self.rng = rng or global_rng.get()

    def __repr__(self) -> str:
        return f"{__class__.__name__}(pmf={self.pmf.tolist()})"

    @cached_property
    def cardinality(self) -> int:
        r"""
        The cardinality $|\mathcal{X}|$ of the source alphabet.
        """
        return self.pmf.size

    def entropy(self, base: LogBase = 2.0) -> float:
        r"""
        Returns the source entropy $\mathrm{H}(X)$. See [`komm.entropy`](/ref/entropy) for more details.

        Parameters:
            base: See [`komm.entropy`](/ref/entropy). The default value is $2.0$.

        Examples:
            >>> dms = komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
            >>> dms.entropy()
            np.float64(1.75)
            >>> dms.entropy(base=4)
            np.float64(0.875)
        """
        return entropy(self.pmf, base)

    def __call__(self, shape: int | tuple[int, ...] = ()) -> npt.NDArray[np.integer]:
        r"""
        Returns random samples from the source.

        Parameters:
            shape: The shape of the output array. If `shape` is an integer, the output array will have shape `(shape,)`. The default value is `()`, which returns a single sample.

        Returns:
            An array of shape `shape` with random samples from the source.

        Examples:
            >>> dms = komm.DiscreteMemorylessSource([0.5, 0.4, 0.1])
            >>> dms()
            array(1)
            >>> dms(10)
            array([0, 1, 1, 0, 2, 1, 1, 0, 0, 0])
            >>> dms((2, 5))
            array([[2, 1, 1, 0, 0],
                   [1, 0, 1, 1, 1]])
        """
        return self.rng.choice(self.pmf.size, p=self.pmf, size=shape)
