import numpy as np
from attrs import field, frozen

from .DiscreteMemorylessChannel import DiscreteMemorylessChannel


@frozen
class BinaryErasureChannel(DiscreteMemorylessChannel):
    r"""
    Binary erasure channel (BEC). It is a [discrete memoryless channel](/ref/DiscreteMemorylessChannel) with input alphabet $\mathcal{X} = \\{ 0, 1 \\}$, output alphabet $\mathcal{Y} = \\{ 0, 1, 2 \\}$, and transition probability matrix given by
    $$
        p_{Y \mid X} =
        \begin{bmatrix}
            1 - \epsilon & 0 & \epsilon \\\\
            0 & 1 - \epsilon & \epsilon
        \end{bmatrix},
    $$
    where the parameter $\epsilon$ is called the *erasure probability* of the channel. For more details, see <cite>CT06, Sec. 7.1.5</cite>.

    Attributes:
        erasure_probability (Optional[float]): The channel erasure probability $\epsilon$. Must satisfy $0 \leq \epsilon \leq 1$. Default value is `0.0`, which corresponds to a noiseless channel.

    Parameters: Input:
        in0 (Array1D[int]): The input sequence.

    Parameters: Output:
        out0 (Array1D[int]): The output sequence.

    Examples:
        >>> np.random.seed(1)
        >>> bec = komm.BinaryErasureChannel(0.1)
        >>> bec.transition_matrix
        array([[0.9, 0. , 0.1],
               [0. , 0.9, 0.1]])
        >>> x = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
        >>> y = bec(x)
        >>> y
        array([1, 1, 2, 0, 0, 2, 1, 0, 1, 0])
    """

    transition_matrix: None = field(init=False, repr=False)
    erasure_probability: float = field(default=0.0)

    def __attrs_post_init__(self):
        epsilon = self.erasure_probability
        tm = np.array([[1 - epsilon, 0, epsilon], [0, 1 - epsilon, epsilon]])
        object.__setattr__(self, "transition_matrix", tm)

    def capacity(self):
        r"""
        Returns the channel capacity $C$. It is given by $C = 1 - \epsilon$. See <cite>CT06, Sec. 7.1.5</cite>.

        Examples:
            >>> bec = komm.BinaryErasureChannel(0.25)
            >>> bec.capacity()
            0.75
        """
        return 1.0 - self.erasure_probability

    def __call__(self, input_sequence):
        epsilon = self.erasure_probability
        erasure_pattern = np.random.rand(np.size(input_sequence)) < epsilon
        output_sequence = np.copy(input_sequence)
        output_sequence[erasure_pattern] = 2
        return output_sequence
