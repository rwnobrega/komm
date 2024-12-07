from typing import Any

import numpy as np
import numpy.typing as npt
from attrs import frozen

from .. import abc
from .._util.information_theory import (
    PMF,
    LogBase,
    assert_is_probability,
    binary_entropy,
)


@frozen
class ZChannel(abc.DiscreteMemorylessChannel):
    r"""
    Z-channel. It is a [discrete memoryless channel](/ref/DiscreteMemorylessChannel) with input and output alphabets $\mathcal{X} = \mathcal{Y} = \\{ 0, 1 \\}$. The channel is characterized by a parameter $p$, called the *decay probability*. Bit $0$ is always received correctly, but bit $1$ turns into $0$ with probability $p$. Equivalently, the channel can be modeled as
    $$
        Y_n = A_n X_n,
    $$
    where $A_n$ are iid Bernoulli random variables with $\Pr[A_n = 0] = p$.

    Attributes:
        decay_probability (Optional[float]): The channel decay probability $p$. Must satisfy $0 \leq p \leq 1$. The default value is `0.0`, which corresponds to a noiseless channel.

    Parameters: Input:
        input_sequence (Array1D[int]): The input sequence.

    Parameters: Output:
        output_sequence (Array1D[int]): The output sequence.

    Examples:
        >>> np.random.seed(1)
        >>> zc = komm.ZChannel(0.1)
        >>> zc([0, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    """

    decay_probability: float = 0.0

    def __attrs_post_init__(self) -> None:
        assert_is_probability(self.decay_probability)

    @property
    def input_cardinality(self) -> int:
        return 2

    @property
    def output_cardinality(self) -> int:
        return 2

    @property
    def transition_matrix(self) -> npt.NDArray[np.float64]:
        r"""
        The transition probability matrix of the channel. It is given by
        $$
            p_{Y \mid X} = \begin{bmatrix} 1 & 0 \\\\ p & 1-p \end{bmatrix}.
        $$

        Examples:
            >>> zc = komm.ZChannel(0.1)
            >>> zc.transition_matrix
            array([[1. , 0. ],
                   [0.1, 0.9]])
        """
        p = self.decay_probability
        return np.array([[1, 0], [p, 1 - p]])

    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        r"""
        Returns the mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$ of the channel. It is given by
        $$
            \mathrm{I}(X ; Y) = \Hb \( \pi (1-p) \) - \pi \Hb(p),
        $$
        in bits, where $\pi = \Pr[X = 1]$, and $\Hb$ is the [binary entropy function](/ref/binary_entropy).

        **Parameters:**

        Same as the [corresponding method](/ref/DiscreteMemorylessChannel/#mutual_information) of the general class.

        Examples:
            >>> zc = komm.ZChannel(0.1)
            >>> zc.mutual_information([0.5, 0.5])  # doctest: +NUMBER
            np.float64(0.7582766571931676)
        """
        input_pmf = PMF(input_pmf)
        p = self.decay_probability
        pi = input_pmf[1]
        return (binary_entropy(pi * (1 - p)) - pi * binary_entropy(p)) / np.log2(base)

    def capacity(self, base: LogBase = 2.0, **kwargs: Any) -> float:
        r"""
        Returns the channel capacity $C$. It is given by
        $$
            C = \log_2 \( 1 + (1-p) p^{p / (1-p)} \),
        $$
        in bits.

        Examples:
            >>> zc = komm.ZChannel(0.1)
            >>> zc.capacity()  # doctest: +NUMBER
            np.float64(0.7628482520105094)
        """
        p = self.decay_probability
        if p == 1.0:
            return 0.0
        q = 1 - p
        return np.log2(1 + q * p ** (p / q)) / np.log2(base)

    def __call__(self, input_sequence: npt.ArrayLike):
        p = self.decay_probability
        input_sequence = np.array(input_sequence)
        keep_pattern = (np.random.rand(np.size(input_sequence)) > p).astype(int)
        return input_sequence * keep_pattern
