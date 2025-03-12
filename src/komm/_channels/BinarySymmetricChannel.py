from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt

from .._util.information_theory import (
    PMF,
    LogBase,
    assert_is_probability,
    binary_entropy,
)
from . import base


@dataclass
class BinarySymmetricChannel(base.DiscreteMemorylessChannel):
    r"""
    Binary symmetric channel (BSC). It is a [discrete memoryless channel](/ref/DiscreteMemorylessChannel) with input and output alphabets $\mathcal{X} = \mathcal{Y} = \\{ 0, 1 \\}$. The channel is characterized by a parameter $p$, called the *crossover probability*. With probability $1 - p$, the output symbol is identical to the input symbol, and with probability $p$, the output symbol is flipped. Equivalently, the channel can be modeled as
    $$
        Y_n = X_n + Z_n,
    $$
    where $Z_n$ are iid Bernoulli random variables with $\Pr[Z_n = 1] = p$. For more details, see <cite>CT06, Sec. 7.1.4</cite>.

    Attributes:
        crossover_probability: The channel crossover probability $p$. Must satisfy $0 \leq p \leq 1$. The default value is `0.0`, which corresponds to a noiseless channel.
    """

    crossover_probability: float = 0.0
    rng: np.random.Generator = field(default=np.random.default_rng(), repr=False)

    def __post_init__(self) -> None:
        assert_is_probability(self.crossover_probability)

    @cached_property
    def input_cardinality(self) -> int:
        r"""
        For the BSC, it is given by $|\mathcal{X}| = 2$.
        """
        return 2

    @cached_property
    def output_cardinality(self) -> int:
        r"""
        For the BSC, it is given by $|\mathcal{Y}| = 2$.
        """
        return 2

    @cached_property
    def transition_matrix(self) -> npt.NDArray[np.floating]:
        r"""
        For the BSC, it is given by
        $$
            p_{Y \mid X} = \begin{bmatrix} 1-p & p \\\\ p & 1-p \end{bmatrix}.
        $$

        Examples:
            >>> bsc = komm.BinarySymmetricChannel(0.2)
            >>> bsc.transition_matrix
            array([[0.8, 0.2],
                   [0.2, 0.8]])
        """
        p = self.crossover_probability
        return np.array([[1 - p, p], [p, 1 - p]])

    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        r"""
        For the BSC, it is given by
        $$
            \mathrm{I}(X ; Y) = \Hb(p + \pi - 2 p \pi) - \Hb(p),
        $$
        in bits, where $\pi = \Pr[X = 1]$, and $\Hb$ is the [binary entropy function](/ref/binary_entropy).

        Examples:
            >>> bsc = komm.BinarySymmetricChannel(0.2)
            >>> bsc.mutual_information([0.45, 0.55])
            np.float64(0.2754734936803773)
        """
        input_pmf = PMF(input_pmf)
        p = self.crossover_probability
        pi = input_pmf[1]
        return (binary_entropy(p + pi - 2 * p * pi) - binary_entropy(p)) / np.log2(base)

    def capacity(self, base: LogBase = 2.0) -> float:
        r"""
        For the BSC, it is given by
        $$
            C = 1 - \Hb(p),
        $$
        in bits, where $\Hb$ is the [binary entropy function](/ref/binary_entropy).

        Examples:
            >>> bsc = komm.BinarySymmetricChannel(0.2)
            >>> bsc.capacity()
            np.float64(0.2780719051126377)
        """
        p = self.crossover_probability
        return (1.0 - binary_entropy(p)) / np.log2(base)

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> rng = np.random.default_rng(seed=42)
            >>> bsc = komm.BinarySymmetricChannel(0.2, rng=rng)
            >>> bsc([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
            array([1, 1, 1, 0, 1, 0, 1, 0, 0, 0])
        """
        p = self.crossover_probability
        input = np.array(input)
        error_pattern = self.rng.random(input.shape) < p
        return (input + error_pattern) % 2
