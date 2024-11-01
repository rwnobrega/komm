import numpy as np

from .DiscreteMemorylessChannel import DiscreteMemorylessChannel


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

    To invoke the channel, call the object giving the input signal as parameter (see example in the constructor below).
    """

    def __init__(self, erasure_probability=0.0):
        r"""
        Constructor for the class.

        Parameters:

            erasure_probability (Optional[float]): The channel erasure probability $\epsilon$. Must satisfy $0 \leq \epsilon \leq 1$. Default value is `0.0`, which corresponds to a noiseless channel.

        Examples:

            >>> np.random.seed(1)
            >>> bec = komm.BinaryErasureChannel(0.1)
            >>> x = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
            >>> y = bec(x); y
            array([1, 1, 2, 0, 0, 2, 1, 0, 1, 0])
        """
        self.erasure_probability = erasure_probability

    @property
    def erasure_probability(self):
        r"""
        The erasure probability $\epsilon$ of the channel.
        """
        return self._erasure_probability

    @erasure_probability.setter
    def erasure_probability(self, value):
        self._erasure_probability = e = float(value)
        self.transition_matrix = [[1 - e, 0, e], [0, 1 - e, e]]

    def capacity(self):
        r"""
        Returns the channel capacity $C$. It is given by $C = 1 - \epsilon$. See <cite>CT06, Sec. 7.1.5</cite>.

        Examples:

            >>> bec = komm.BinaryErasureChannel(0.25)
            >>> bec.capacity()
            0.75
        """
        return 1.0 - self._erasure_probability

    def __call__(self, input_sequence):
        erasure_pattern = (
            np.random.rand(np.size(input_sequence)) < self._erasure_probability
        )
        output_sequence = np.copy(input_sequence)
        output_sequence[erasure_pattern] = 2
        return output_sequence

    def __repr__(self):
        args = "erasure_probability={}".format(self._erasure_probability)
        return "{}({})".format(self.__class__.__name__, args)
