import numpy as np
from attrs import field, frozen

from .._util.information_theory import _arimoto_blahut, _mutual_information


@frozen
class DiscreteMemorylessChannel:
    r"""
    Discrete memoryless channel (DMC). It is defined by an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, and a *transition probability matrix* $p_{Y \mid X}$. Here, for simplicity, the input and output alphabets are always taken as $\mathcal{X} = \\{ 0, 1, \ldots, |\mathcal{X}| - 1 \\}$ and $\mathcal{Y} = \\{ 0, 1, \ldots, |\mathcal{Y}| - 1 \\}$, respectively. The transition probability matrix $p_{Y \mid X}$, of size $|\mathcal{X}|$-by-$|\mathcal{Y}|$, gives the conditional probability of receiving $Y = y$ given that $X = x$ is transmitted. For more details, see <cite>CT06, Ch. 7</cite>.

    To invoke the channel, call the object giving the input signal as parameter (see example below).

    Attributes:
        transition_matrix (Array2D[float]): The channel transition probability matrix $p_{Y \mid X}$. The element in row $x \in \mathcal{X}$ and column $y \in \mathcal{Y}$ must be equal to $p_{Y \mid X}(y \mid x)$.

    Examples:
        >>> np.random.seed(1)
        >>> dmc = komm.DiscreteMemorylessChannel([[0.9, 0.05, 0.05], [0.0, 0.5, 0.5]])
        >>> x = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1]
        >>> y = dmc(x); y
        array([0, 2, 0, 1, 1, 1, 0, 0, 0, 2])
    """

    transition_matrix: np.ndarray = field(converter=np.asarray)

    @property
    def input_cardinality(self):
        r"""
        The channel input cardinality $|\mathcal{X}|$.
        """
        return self.transition_matrix.shape[0]

    @property
    def output_cardinality(self):
        r"""
        The channel output cardinality $|\mathcal{Y}|$.
        """
        return self.transition_matrix.shape[1]

    def mutual_information(self, input_pmf, base=2.0):
        r"""
        Computes the mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$ of the channel. It is given by
        $$
            \mathrm{I}(X ; Y) = \mathrm{H}(X) - \mathrm{H}(X \mid Y),
        $$
        where $\mathrm{H}(X)$ is the the entropy of $X$ and $\mathrm{H}(X \mid Y)$ is the conditional entropy of $X$ given $Y$. By default, the base of the logarithm is $2$, in which case the mutual information is measured in bits. See <cite>CT06, Ch. 2</cite>.

        Parameters:
            input_pmf (Array1D[float]): The probability mass function $p_X$ of the channel input $X$. It must be a valid pmf, that is, all of its values must be non-negative and sum up to $1$.

            base (Optional[float | str]): The base of the logarithm to be used. It must be a positive float or the string `'e'`. The default value is `2.0`.

        Returns:
            mutual_information (float): The mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$.

        Examples:
            >>> dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2], [0.5, 0.05, 0.45]])
            >>> dmc.mutual_information([1/3, 1/3, 1/3])  # doctest: +NUMBER
            np.float64(0.123811098798)
            >>> dmc.mutual_information([1/3, 1/3, 1/3], base=3)  # doctest: +NUMBER
            np.float64(0.078116106054)
        """
        return _mutual_information(input_pmf, self.transition_matrix, base)

    def capacity(self, base=2.0, arimoto_blahut_kwargs=None):
        r"""
        Returns the channel capacity $C$. It is given by $C = \max_{p_X} \mathrm{I}(X;Y)$. This method computes the channel capacity via the Arimoto–Blahut algorithm. See <cite>CT06, Sec. 10.8</cite>.

        Parameters:
            base (Optional[float | str]): The base of the logarithm to be used. It must be a positive float or the string `'e'`. The default value is `2.0`.

        Returns:
            capacity (float): The channel capacity $C$.

        Examples:
            >>> dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2], [0.5, 0.05, 0.45]])
            >>> dmc.capacity()  # doctest: +NUMBER
            np.float64(0.1616318610)
            >>> dmc.capacity(base=3)  # doctest: +NUMBER
            np.float64(0.1019783502)
        """
        if arimoto_blahut_kwargs is None:
            arimoto_blahut_kwargs = {"max_iters": 1000, "error_tolerance": 1e-12}
        initial_guess = (
            np.ones(self.input_cardinality, dtype=float) / self.input_cardinality
        )
        optimal_input_pmf = _arimoto_blahut(
            self.transition_matrix, initial_guess, **arimoto_blahut_kwargs
        )
        return _mutual_information(optimal_input_pmf, self.transition_matrix, base=base)

    def __call__(self, input_sequence):
        output_sequence = [
            np.random.choice(
                self.output_cardinality, p=self.transition_matrix[input_symbol]
            )
            for input_symbol in input_sequence
        ]
        return np.array(output_sequence)
