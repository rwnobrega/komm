from typing import Any, cast

import numpy as np
import numpy.typing as npt
from attr import field
from attrs import field, frozen

from .._util.information_theory import LogBase, binary_entropy
from .._validation import is_pmf, is_probability, validate_call
from .AbstractDiscreteMemorylessChannel import AbstractDiscreteMemorylessChannel


@frozen
class BinaryErasureChannel(AbstractDiscreteMemorylessChannel):
    r"""
    Binary erasure channel (BEC). It is a [discrete memoryless channel](/ref/DiscreteMemorylessChannel) with input alphabet $\mathcal{X} = \\{ 0, 1 \\}$ and output alphabet $\mathcal{Y} = \\{ 0, 1, 2 \\}$. The channel is characterized by a parameter $\epsilon$, called the *erasure probability*. With probability $1 - \epsilon$, the output symbol is identical to the input symbol, and with probability $\epsilon$, the output symbol is replaced by an erasure symbol (denoted by $2$). For more details, see <cite>CT06, Sec. 7.1.5</cite>.

    Attributes:
        erasure_probability (Optional[float]): The channel erasure probability $\epsilon$. Must satisfy $0 \leq \epsilon \leq 1$. Default value is `0.0`, which corresponds to a noiseless channel.

    Parameters: Input:
        input_sequence (Array1D[int]): The input sequence.

    Parameters: Output:
        output_sequence (Array1D[int]): The output sequence.

    Examples:
        >>> np.random.seed(1)
        >>> bec = komm.BinaryErasureChannel(0.1)
        >>> bec([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        array([1, 1, 2, 0, 0, 2, 1, 0, 1, 0])
    """

    erasure_probability: float = field(default=0.0, validator=is_probability)

    @property
    def input_cardinality(self) -> int:
        return 2

    @property
    def output_cardinality(self) -> int:
        return 3

    @property
    def transition_matrix(self) -> npt.NDArray[np.float64]:
        r"""
        The transition probability matrix of the channel, given by
        $$
            p_{Y \mid X} =
            \begin{bmatrix}
                1 - \epsilon & 0 & \epsilon \\\\
                0 & 1 - \epsilon & \epsilon
            \end{bmatrix}.
        $$

        Examples:
            >>> bec = komm.BinaryErasureChannel(0.1)
            >>> bec.transition_matrix
            array([[0.9, 0. , 0.1],
                   [0. , 0.9, 0.1]])
        """
        epsilon = self.erasure_probability
        return np.array([[1 - epsilon, 0, epsilon], [0, 1 - epsilon, epsilon]])

    @validate_call(input_pmf=field(converter=np.asarray, validator=is_pmf))
    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        r"""
        Returns the mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$ of the channel. It is given by
        $$
            \mathrm{I}(X ; Y) = (1 - \epsilon) \, \Hb(\pi),
        $$
        in bits, where $\pi = \Pr[X = 1]$, and $\Hb$ is the [binary entropy function](/ref/binary_entropy).

        **Parameters:**

        Same as the [corresponding method](/ref/DiscreteMemorylessChannel/#mutual_information) of the general class.

        Examples:
            >>> bec = komm.BinaryErasureChannel(0.1)
            >>> bec.mutual_information([0.45, 0.55]) # doctest: +NUMBER
            np.float64(0.8934970085890275)
        """
        input_pmf = cast(npt.NDArray[np.float64], input_pmf)
        epsilon = self.erasure_probability
        pi = input_pmf[1]
        return (1.0 - epsilon) * binary_entropy(pi) / np.log2(base)

    def capacity(self, base: LogBase = 2.0, **kwargs: Any) -> float:
        r"""
        Returns the channel capacity $C$. It is given by
        $$
            C = 1 - \epsilon,
        $$
        in bits.

        Examples:
            >>> bec = komm.BinaryErasureChannel(0.1)
            >>> bec.capacity()
            np.float64(0.9)
        """
        return (1.0 - self.erasure_probability) / np.log2(base)

    def __call__(self, input_sequence: npt.ArrayLike) -> npt.NDArray[np.int_]:
        epsilon = self.erasure_probability
        erasure_pattern = np.random.rand(np.size(input_sequence)) < epsilon
        output_sequence = np.copy(input_sequence)
        output_sequence[erasure_pattern] = 2
        return output_sequence
