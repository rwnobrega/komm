import numpy as np

from .._util import _entropy
from .DiscreteMemorylessChannel import DiscreteMemorylessChannel


class BinarySymmetricChannel(DiscreteMemorylessChannel):
    r"""
    Binary symmetric channel (BSC). It is a [discrete memoryless channel](/ref/DiscreteMemorylessChannel) with input and output alphabets given by $\mathcal{X} = \mathcal{Y} = \\{ 0, 1 \\}$, and transition probability matrix given by
    $$
        p_{Y \mid X} = \begin{bmatrix} 1-p & p \\\\ p & 1-p \end{bmatrix},
    $$
    where the parameter $p$ is called the *crossover probability* of the channel. Equivalently, a BSC with crossover probability $p$ may be defined by
    $$
        Y_n = X_n + Z_n,
    $$
    where $Z_n$ are iid Bernoulli random variables with $\Pr[Z_n = 1] = p$. For more details, see <cite>CT06, Sec. 7.1.4</cite>.

    To invoke the channel, call the object giving the input signal as parameter (see example below).
    """

    def __init__(self, crossover_probability=0.0):
        r"""
        Constructor for the class.

        Parameters:

            crossover_probability (Optional[float]): The channel crossover probability $p$. Must satisfy $0 \leq p \leq 1$. The default value is `0.0`, which corresponds to a noiseless channel.

        Examples:

            >>> np.random.seed(1)
            >>> bsc = komm.BinarySymmetricChannel(0.1)
            >>> x = [0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
            >>> y = bsc(x); y
            array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1])
        """
        self.crossover_probability = crossover_probability

    @property
    def crossover_probability(self):
        r"""
        The crossover probability $p$ of the channel.
        """
        return self._crossover_probability

    @crossover_probability.setter
    def crossover_probability(self, value):
        self._crossover_probability = p = float(value)
        self.transition_matrix = np.array([[1 - p, p], [p, 1 - p]])

    def capacity(self):
        r"""
        Returns the channel capacity $C$. It is given by $C = 1 - \mathcal{H}(p)$. See <cite>CT06, Sec. 7.1.4</cite>.

        Examples:

            >>> bsc = komm.BinarySymmetricChannel(0.25)
            >>> bsc.capacity()
            np.float64(0.18872187554086717)
        """
        return 1.0 - _entropy(
            np.array([self._crossover_probability, 1.0 - self._crossover_probability]),
            2.0,
        )

    def __call__(self, input_sequence):
        error_pattern = (
            np.random.rand(np.size(input_sequence)) < self._crossover_probability
        ).astype(int)
        return (input_sequence + error_pattern) % 2

    def __repr__(self):
        args = "crossover_probability={}".format(self._crossover_probability)
        return "{}({})".format(self.__class__.__name__, args)
