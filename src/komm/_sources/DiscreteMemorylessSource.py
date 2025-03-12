from functools import cached_property

import numpy as np
import numpy.typing as npt

from .._util.information_theory import PMF, LogBase, entropy


class DiscreteMemorylessSource:
    r"""
    Discrete memoryless source (DMS). It is defined by an *alphabet* $\mathcal{X}$ and a *probability mass function* (pmf) $p_X$. Here, for simplicity, the alphabet is always taken as $\mathcal{X} = \\{ 0, 1, \ldots, |\mathcal{X}| - 1 \\}$. The pmf $p_X$ gives the probability of the source emitting the symbol $X = x$.

    Parameters:
        pmf: The source probability mass function $p_X$. The element in position $x \in \mathcal{X}$ must be equal to $p_X(x)$.
    """

    def __init__(
        self, pmf: npt.ArrayLike, rng: np.random.Generator = np.random.default_rng()
    ):
        self.pmf = PMF(pmf)
        self.rng = rng

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
            >>> rng = np.random.default_rng(seed=42)
            >>> dms = komm.DiscreteMemorylessSource([0.5, 0.4, 0.1], rng=rng)
            >>> dms()
            array(1)
            >>> dms(10)
            array([0, 1, 1, 0, 2, 1, 1, 0, 0, 0])
            >>> dms((2, 5))
            array([[2, 1, 1, 0, 0],
                   [1, 0, 1, 1, 1]])
        """
        return self.rng.choice(self.pmf.size, p=self.pmf, size=shape)
