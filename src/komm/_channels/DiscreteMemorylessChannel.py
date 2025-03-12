from functools import cached_property

import numpy as np
import numpy.typing as npt

from .._util.information_theory import (
    LogBase,
    TransitionMatrix,
    arimoto_blahut,
    mutual_information,
)
from . import base


class DiscreteMemorylessChannel(base.DiscreteMemorylessChannel):
    r"""
    General discrete memoryless channel (DMC). It is defined by an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, and a *transition probability matrix* $p_{Y \mid X}$. Here, for simplicity, the input and output alphabets are always taken as $\mathcal{X} = \\{ 0, 1, \ldots, |\mathcal{X}| - 1 \\}$ and $\mathcal{Y} = \\{ 0, 1, \ldots, |\mathcal{Y}| - 1 \\}$, respectively. The transition probability matrix $p_{Y \mid X}$, of size $|\mathcal{X}|$-by-$|\mathcal{Y}|$, gives the conditional probability of receiving $Y = y$ given that $X = x$ is transmitted. For more details, see <cite>CT06, Ch. 7</cite>.

    Attributes:
        transition_matrix: The channel transition probability matrix $p_{Y \mid X}$. The element in row $x \in \mathcal{X}$ and column $y \in \mathcal{Y}$ must be equal to $p_{Y \mid X}(y \mid x)$.
    """

    def __init__(
        self,
        transition_matrix: npt.ArrayLike,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self._transition_matrix = TransitionMatrix(transition_matrix)
        self.rng = rng

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.transition_matrix.tolist()})"

    @cached_property
    def input_cardinality(self) -> int:
        return self.transition_matrix.shape[0]

    @cached_property
    def output_cardinality(self) -> int:
        return self.transition_matrix.shape[1]

    @cached_property
    def transition_matrix(self) -> npt.NDArray[np.floating]:
        return self._transition_matrix

    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        r"""
        The mutual information is given by
        $$
            \mathrm{I}(X; Y) = \mathrm{H}(X) - \mathrm{H}(X \mid Y),
        $$
        where $\mathrm{H}(X)$ is the the entropy of $X$ and $\mathrm{H}(X \mid Y)$ is the conditional entropy of $X$ given $Y$. By default, the base of the logarithm is $2$, in which case the mutual information is measured in bits. See <cite>CT06, Ch. 2</cite>.

        Examples:
            >>> dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2], [0.5, 0.05, 0.45]])
            >>> dmc.mutual_information([1/3, 1/3, 1/3]).round(6)
            np.float64(0.123811)
            >>> dmc.mutual_information([1/3, 1/3, 1/3], base=3).round(6)
            np.float64(0.078116)
            >>> dmc.mutual_information([1/3, 1/3, 1/3], base='e').round(6)
            np.float64(0.085819)
        """
        return mutual_information(input_pmf, self.transition_matrix, base)

    def capacity(self, base: LogBase = 2.0) -> float:
        r"""
        The channel capacity is given by
        $$
            C = \max_{p_X} \mathrm{I}(X; Y).
        $$
        This method computes the channel capacity via the Arimoto–Blahut algorithm. See <cite>CT06, Sec. 10.8</cite>.

        Examples:
            >>> dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2], [0.5, 0.05, 0.45]])
            >>> dmc.capacity().round(6)
            np.float64(0.161632)
            >>> dmc.capacity(base=3).round(6)
            np.float64(0.101978)
            >>> dmc.capacity(base='e').round(6)
            np.float64(0.112035)
        """
        initial_guess = np.ones(self.input_cardinality) / self.input_cardinality
        optimal_input_pmf = arimoto_blahut(
            self.transition_matrix, initial_guess, max_iter=1000, tol=1e-12
        )
        return mutual_information(optimal_input_pmf, self.transition_matrix, base=base)

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> rng = np.random.default_rng(seed=42)
            >>> dmc = komm.DiscreteMemorylessChannel([[0.9, 0.05, 0.05], [0.0, 0.5, 0.5]], rng=rng)
            >>> dmc([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
            array([2, 1, 2, 0, 0, 2, 2, 0, 1, 0])
        """
        input = np.asarray(input)
        output = np.empty_like(input, dtype=int)
        for index, symbol in np.ndenumerate(input):
            output[index] = self.rng.choice(
                a=self.output_cardinality, p=self.transition_matrix[symbol]
            )
        return output
