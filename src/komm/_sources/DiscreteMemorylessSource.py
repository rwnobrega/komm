from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._util import global_rng
from .._util.information_theory import entropy
from .._util.validators import validate_pmf
from ..types import Array1D


class DiscreteMemorylessSource:
    r"""
    Discrete memoryless source. It is defined by a finite *source alphabet* $\mathcal{X}$ and a *pmf* $p$ over $\mathcal{X}$. The value of $p(x)$ gives the probability of the source emitting symbol $x \in \mathcal{X}$, with the probability of emitting a symbol being independent of all previously emitted symbols. Here, for simplicity, the alphabet is taken as $\mathcal{X} = [0 : |\mathcal{X}|)$, where $|\mathcal{X}|$ is called the *source cardinality*.

    Parameters:
        pmf: Either a one-dimensional array of floats representing the source pmf $p$, or a positive integer $M$, in which case a uniform pmf over $[0 : M)$ is assumed.

    Note:
        The cardinality $|\mathcal{X}|$ is inferred from the length of the `pmf` array.
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
        self._pmf = validate_pmf(pmf, joint=False)
        self._rng = rng or global_rng.get()

    def __repr__(self) -> str:
        return f"{__class__.__name__}(pmf={self.pmf.tolist()})"

    @cached_property
    def pmf(self) -> Array1D[np.floating]:
        r"""
        The source pmf $p$ over $\mathcal{X}$.

        Examples:
            >>> source = komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
            >>> source.pmf
            array([0.5  , 0.25 , 0.125, 0.125])

            >>> source = komm.DiscreteMemorylessSource(4)
            >>> source.pmf
            array([0.25, 0.25, 0.25, 0.25])
        """
        return self._pmf

    @cached_property
    def cardinality(self) -> int:
        r"""
        The source cardinality $|\mathcal{X}|$.

        Examples:
            >>> source = komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
            >>> source.cardinality
            4

            >>> source = komm.DiscreteMemorylessSource(4)
            >>> source.cardinality
            4
        """
        return self.pmf.size

    def entropy_rate(self, base: float | Literal["e"] = 2.0) -> np.floating:
        r"""
        Computes the source entropy rate. For a discrete memoryless source, this is simply the [entropy](/ref/entropy) of the pmf $p$.

        Parameters:
            base: See [`komm.entropy`](/ref/entropy). The default value is `2.0`.

        Returns:
            entropy_rate: The source entropy rate.

        Examples:
            >>> source = komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
            >>> source.entropy_rate()
            np.float64(1.75)
            >>> source.entropy_rate(base=4)
            np.float64(0.875)

            >>> source = komm.DiscreteMemorylessSource(4)
            >>> source.entropy_rate()
            np.float64(2.0)
            >>> source.entropy_rate(base=4)
            np.float64(1.0)
        """
        return entropy(self.pmf, base)

    def emit(
        self,
        shape: tuple[int, ...] | int | None = None,
    ) -> npt.NDArray[np.integer]:
        r"""
        Returns random symbols from the source.

        Parameters:
            shape: The shape of the output array. The default value corresponds to a single symbol.

        Returns:
            symbols: The emitted symbols from the source. It is an array with elements in $\mathcal{X}$ and the given shape.

        Examples:
            >>> source = komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
            >>> source.emit()
            array([2])
            >>> source.emit(10)
            array([0, 2, 1, 0, 3, 2, 2, 0, 0, 0])
            >>> source.emit((2, 5))
            array([[3, 1, 2, 0, 0],
                   [1, 0, 2, 1, 2]])
        """
        if shape is None:
            shape = (1,)
        elif isinstance(shape, int):
            shape = (shape,)
        return self._rng.choice(self.pmf.size, p=self.pmf, size=shape)
