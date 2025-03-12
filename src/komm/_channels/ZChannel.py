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
class ZChannel(base.DiscreteMemorylessChannel):
    r"""
    Z-channel. It is a [discrete memoryless channel](/ref/DiscreteMemorylessChannel) with input and output alphabets $\mathcal{X} = \mathcal{Y} = \\{ 0, 1 \\}$. The channel is characterized by a parameter $p$, called the *decay probability*. Bit $0$ is always received correctly, but bit $1$ turns into $0$ with probability $p$. Equivalently, the channel can be modeled as
    $$
        Y_n = A_n X_n,
    $$
    where $A_n$ are iid Bernoulli random variables with $\Pr[A_n = 0] = p$.

    Attributes:
        decay_probability: The channel decay probability $p$. Must satisfy $0 \leq p \leq 1$. The default value is `0.0`, which corresponds to a noiseless channel.
    """

    decay_probability: float = 0.0
    rng: np.random.Generator = field(default=np.random.default_rng(), repr=False)

    def __post_init__(self) -> None:
        assert_is_probability(self.decay_probability)

    @cached_property
    def input_cardinality(self) -> int:
        r"""
        For the Z-channel, it is given by $|\mathcal{X}| = 2$.
        """
        return 2

    @cached_property
    def output_cardinality(self) -> int:
        r"""
        For the Z-channel, it is given by $|\mathcal{Y}| = 2$.
        """
        return 2

    @cached_property
    def transition_matrix(self) -> npt.NDArray[np.floating]:
        r"""
        For the Z-channel, it is given by
        $$
            p_{Y \mid X} = \begin{bmatrix} 1 & 0 \\\\ p & 1-p \end{bmatrix}.
        $$

        Examples:
            >>> zc = komm.ZChannel(0.2)
            >>> zc.transition_matrix
            array([[1. , 0. ],
                   [0.2, 0.8]])
        """
        p = self.decay_probability
        return np.array([[1, 0], [p, 1 - p]])

    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        r"""
        For the Z-channel, it is given by
        $$
            \mathrm{I}(X ; Y) = \Hb \( \pi (1-p) \) - \pi \Hb(p),
        $$
        in bits, where $\pi = \Pr[X = 1]$, and $\Hb$ is the [binary entropy function](/ref/binary_entropy).

        Examples:
            >>> zc = komm.ZChannel(0.2)
            >>> zc.mutual_information([0.5, 0.5])  # doctest: +FLOAT_CMP
            np.float64(0.6099865470109874)
        """
        input_pmf = PMF(input_pmf)
        p = self.decay_probability
        pi = input_pmf[1]
        return (binary_entropy(pi * (1 - p)) - pi * binary_entropy(p)) / np.log2(base)

    def capacity(self, base: LogBase = 2.0) -> float:
        r"""
        For the Z-channel, it is given by
        $$
            C = \log_2 \( 1 + (1-p) p^{p / (1-p)} \),
        $$
        in bits.

        Examples:
            >>> zc = komm.ZChannel(0.2)
            >>> zc.capacity()  # doctest: +FLOAT_CMP
            np.float64(0.6182313659549211)
        """
        p = self.decay_probability
        if p == 1.0:
            return 0.0
        q = 1 - p
        return np.log2(1 + q * p ** (p / q)) / np.log2(base)

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> rng = np.random.default_rng(seed=42)
            >>> zc = komm.ZChannel(0.2, rng=rng)
            >>> zc([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
            array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
        """
        p = self.decay_probability
        input = np.asarray(input)
        keep_pattern = self.rng.random(input.shape) >= p
        return (input * keep_pattern).astype(int)
