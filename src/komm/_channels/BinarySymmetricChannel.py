from typing import Any

import numpy as np
import numpy.typing as npt
from attrs import field, frozen

from .._util.information_theory import LogBase, binary_entropy
from .._validation import is_pmf, is_probability, validate_call
from .AbstractDiscreteMemorylessChannel import AbstractDiscreteMemorylessChannel


@frozen
class BinarySymmetricChannel(AbstractDiscreteMemorylessChannel):
    r"""
    Binary symmetric channel (BSC). It is a [discrete memoryless channel](/ref/DiscreteMemorylessChannel) with input and output alphabets $\mathcal{X} = \mathcal{Y} = \\{ 0, 1 \\}$. The channel is characterized by a parameter $p$, called the *crossover probability*. With probability $1 - p$, the output symbol is identical to the input symbol, and with probability $p$, the output symbol is flipped. Equivalently, the channel can be modeled as
    $$
        Y_n = X_n + Z_n,
    $$
    where $Z_n$ are iid Bernoulli random variables with $\Pr[Z_n = 1] = p$. For more details, see <cite>CT06, Sec. 7.1.4</cite>.

    Attributes:
        crossover_probability (Optional[float]): The channel crossover probability $p$. Must satisfy $0 \leq p \leq 1$. The default value is `0.0`, which corresponds to a noiseless channel.

    Parameters: Input:
        input_sequence (Array1D[int]): The input sequence.

    Parameters: Output:
        output_sequence (Array1D[int]): The output sequence.

    Examples:
        >>> np.random.seed(1)
        >>> bsc = komm.BinarySymmetricChannel(0.1)
        >>> bsc([0, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1])
    """

    crossover_probability: float = field(default=0.0, validator=is_probability)

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
            p_{Y \mid X} = \begin{bmatrix} 1-p & p \\\\ p & 1-p \end{bmatrix}.
        $$

        Examples:
            >>> bsc = komm.BinarySymmetricChannel(0.1)
            >>> bsc.transition_matrix
            array([[0.9, 0.1],
                   [0.1, 0.9]])
        """
        p = self.crossover_probability
        return np.array([[1 - p, p], [p, 1 - p]])

    @validate_call(input_pmf=field(converter=np.asarray, validator=is_pmf))
    def mutual_information(
        self, input_pmf: npt.ArrayLike, base: LogBase = 2.0
    ) -> float:
        r"""
        Returns the mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$ of the channel. It is given by
        $$
            \mathrm{I}(X ; Y) = \Hb(p + \pi - 2 p \pi) - \Hb(p),
        $$
        in bits, where $\pi = \Pr[X = 1]$, and $\Hb$ is the [binary entropy function](/ref/binary_entropy).

        **Parameters:**

        Same as the [corresponding method](/ref/DiscreteMemorylessChannel/#mutual_information) of the general class.

        Examples:
            >>> bsc = komm.BinarySymmetricChannel(0.1)
            >>> bsc.mutual_information([0.45, 0.55])
            np.float64(0.5263828452309445)
        """
        input_pmf = np.array(input_pmf)
        p = self.crossover_probability
        pi = input_pmf[1]
        return (binary_entropy(p + pi - 2 * p * pi) - binary_entropy(p)) / np.log2(base)

    def capacity(self, base: LogBase = 2.0, **kwargs: Any) -> float:
        r"""
        Returns the channel capacity $C$. It is given by
        $$
            C = 1 - \Hb(p),
        $$
        in bits, where $\Hb$ is the [binary entropy function](/ref/binary_entropy).

        Examples:
            >>> bsc = komm.BinarySymmetricChannel(0.1)
            >>> bsc.capacity()
            np.float64(0.5310044064107188)
        """
        p = self.crossover_probability
        return (1.0 - binary_entropy(p)) / np.log2(base)

    def __call__(self, input_sequence: npt.ArrayLike):
        p = self.crossover_probability
        input_sequence = np.array(input_sequence)
        error_pattern = (np.random.rand(np.size(input_sequence)) < p).astype(int)
        return (input_sequence + error_pattern) % 2
