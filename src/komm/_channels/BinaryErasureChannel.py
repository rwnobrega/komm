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
class BinaryErasureChannel(base.DiscreteMemorylessChannel):
    r"""
    Binary erasure channel (BEC). It is a [discrete memoryless channel](/ref/DiscreteMemorylessChannel) with input alphabet $\mathcal{X} = \\{ 0, 1 \\}$ and output alphabet $\mathcal{Y} = \\{ 0, 1, 2 \\}$. The channel is characterized by a parameter $\epsilon$, called the *erasure probability*. With probability $1 - \epsilon$, the output symbol is identical to the input symbol, and with probability $\epsilon$, the output symbol is replaced by an erasure symbol (denoted by $2$). For more details, see <cite>CT06, Sec. 7.1.5</cite>.

    Attributes:
        erasure_probability: The channel erasure probability $\epsilon$. Must satisfy $0 \leq \epsilon \leq 1$. Default value is `0.0`, which corresponds to a noiseless channel.
    """

    erasure_probability: float = 0.0
    rng: np.random.Generator = field(default=np.random.default_rng(), repr=False)

    def __post_init__(self) -> None:
        assert_is_probability(self.erasure_probability)

    @cached_property
    def input_cardinality(self) -> int:
        r"""
        For the BEC, it is given by $|\mathcal{X}| = 2$.
        """
        return 2

    @cached_property
    def output_cardinality(self) -> int:
        r"""
        For the BEC, it is given by $|\mathcal{Y}| = 3$.
        """
        return 3

    @cached_property
    def transition_matrix(self) -> npt.NDArray[np.floating]:
        r"""
        For the BEC, it is given by
        $$
            p_{Y \mid X} =
            \begin{bmatrix}
                1 - \epsilon & 0 & \epsilon \\\\
                0 & 1 - \epsilon & \epsilon
            \end{bmatrix}.
        $$

        Examples:
            >>> bec = komm.BinaryErasureChannel(0.2)
            >>> bec.transition_matrix
            array([[0.8, 0. , 0.2],
                   [0. , 0.8, 0.2]])
        """
        epsilon = self.erasure_probability
        return np.array([[1 - epsilon, 0, epsilon], [0, 1 - epsilon, epsilon]])

    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        r"""
        For the BEC, it is given by
        $$
            \mathrm{I}(X ; Y) = (1 - \epsilon) \, \Hb(\pi),
        $$
        in bits, where $\pi = \Pr[X = 1]$, and $\Hb$ is the [binary entropy function](/ref/binary_entropy).

        Examples:
            >>> bec = komm.BinaryErasureChannel(0.2)
            >>> bec.mutual_information([0.45, 0.55]) # doctest: +FLOAT_CMP
            np.float64(0.7942195631902467)
        """
        input_pmf = PMF(input_pmf)
        epsilon = self.erasure_probability
        pi = input_pmf[1]
        return (1.0 - epsilon) * binary_entropy(pi) / np.log2(base)

    def capacity(self, base: LogBase = 2.0) -> float:
        r"""
        For the BEC, it is given by
        $$
            C = 1 - \epsilon,
        $$
        in bits.

        Examples:
            >>> bec = komm.BinaryErasureChannel(0.2)
            >>> bec.capacity()
            np.float64(0.8)
        """
        return (1.0 - self.erasure_probability) / np.log2(base)

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> rng = np.random.default_rng(seed=42)
            >>> bec = komm.BinaryErasureChannel(0.2, rng=rng)
            >>> bec([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
            array([1, 1, 1, 0, 2, 0, 1, 0, 2, 0])
        """
        epsilon = self.erasure_probability
        input = np.asarray(input)
        erasure_pattern = self.rng.random(input.shape) < epsilon
        output = np.copy(input)
        output[erasure_pattern] = 2
        return output
