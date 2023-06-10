import numpy as np

from .._util import _mutual_information


class DiscreteMemorylessChannel:
    r"""
    Discrete memoryless channel (DMC). It is defined by an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, and a *transition probability matrix* $p_{Y \mid X}$. Here, for simplicity, the input and output alphabets are always taken as $\mathcal{X} = \\{ 0, 1, \ldots, |\mathcal{X}| - 1 \\}$ and $\mathcal{Y} = \\{ 0, 1, \ldots, |\mathcal{Y}| - 1 \\}$, respectively. The transition probability matrix $p_{Y \mid X}$, of size $|\mathcal{X}|$-by-$|\mathcal{Y}|$, gives the conditional probability of receiving $Y = y$ given that $X = x$ is transmitted. See :cite:`Cover.Thomas.06` (Ch. 7).

    To invoke the channel, call the object giving the input signal as parameter (see example in the constructor below).
    """

    def __init__(self, transition_matrix):
        r"""
        Constructor for the class.

        Parameters:

            transition_matrix (2D-array of :obj:`float`): The channel transition probability matrix $p_{Y \mid X}$. The element in row $x \in \mathcal{X}$ and column $y \in \mathcal{Y}$ must be equal to $p_{Y \mid X}(y \mid x)$.

        Examples:

            >>> dmc = komm.DiscreteMemorylessChannel([[0.9, 0.05, 0.05], [0.0, 0.5, 0.5]])
            >>> x = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1]
            >>> y = dmc(x); y  #doctest: +SKIP
            array([0, 2, 0, 2, 1, 1, 0, 0, 0, 2])
        """
        self.transition_matrix = transition_matrix
        self._arimoto_blahut_kwargs = {"max_iters": 1000, "error_tolerance": 1e-12}

    @property
    def transition_matrix(self):
        r"""
        The channel transition probability matrix $p_{Y \mid X}$. This is a read-and-write property.
        """
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value):
        self._transition_matrix = np.array(value, dtype=float)
        self._input_cardinality, self._output_cardinality = self._transition_matrix.shape

    @property
    def input_cardinality(self):
        r"""
        The channel input cardinality $|\mathcal{X}|$. This property is read-only.
        """
        return self._input_cardinality

    @property
    def output_cardinality(self):
        r"""
        The channel output cardinality $|\mathcal{Y}|$. This property is read-only.
        """
        return self._output_cardinality

    def mutual_information(self, input_pmf, base=2.0):
        r"""
        Computes the mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$ of the channel. It is given by

        .. math::
           \mathrm{I}(X ; Y) = \mathrm{H}(X) - \mathrm{H}(X \mid Y),

        where $\mathrm{H}(X)$ is the the entropy of $X$ and $\mathrm{H}(X \mid Y)$ is the conditional entropy of $X$ given $Y$. By default, the base of the logarithm is $2$, in which case the mutual information is measured in bits. See :cite:`Cover.Thomas.06` (Ch. 2).

        Parameters:

            input_pmf (1D-array of :obj:`float`): The probability mass function $p_X$ of the channel input $X$. It must be a valid :term:`pmf`, that is, all of its values must be non-negative and sum up to $1$.

            base (:obj:`float` or :obj:`str`, optional): The base of the logarithm to be used. It must be a positive float or the string :code:`'e'`. The default value is :code:`2.0`.

        Returns:

            mutual_information (:obj:`float`): The mutual information $\mathrm{I}(X ; Y)$ between the input $X$ and the output $Y$.

        Examples:

            >>> dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2], [0.5, 0.05, 0.45]])
            >>> dmc.mutual_information([1/3, 1/3, 1/3])
            0.12381109879798724
            >>> dmc.mutual_information([1/3, 1/3, 1/3], base=3)
            0.07811610605402552
        """
        return _mutual_information(input_pmf, self._transition_matrix, base)

    def capacity(self, base=2.0):
        r"""
        Returns the channel capacity $C$. It is given by $C = \max_{p_X} \mathrm{I}(X;Y)$. This method computes the channel capacity via the Arimoto--Blahut algorithm. See :cite:`Cover.Thomas.06` (Sec. 10.8).

        Parameters:

            base (:obj:`float` or :obj:`str`, optional): The base of the logarithm to be used. It must be a positive float or the string :code:`'e'`. The default value is :code:`2.0`.

        Returns:

            capacity (:obj:`float`): The channel capacity $C$.

        Examples:

            >>> dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2], [0.5, 0.05, 0.45]])
            >>> dmc.capacity()
            0.1616318609548566
            >>> dmc.capacity(base=3)
            0.10197835020154389
        """
        initial_guess = np.ones(self._input_cardinality, dtype=float) / self._input_cardinality
        optimal_input_pmf = self._arimoto_blahut(self._transition_matrix, initial_guess, **self._arimoto_blahut_kwargs)
        return _mutual_information(optimal_input_pmf, self._transition_matrix, base=base)

    def __call__(self, input_sequence):
        output_sequence = [
            np.random.choice(self._output_cardinality, p=self._transition_matrix[input_symbol])
            for input_symbol in input_sequence
        ]
        return np.array(output_sequence)

    def __repr__(self):
        args = "transition_matrix={}".format(self._transition_matrix.tolist())
        return "{}({})".format(self.__class__.__name__, args)

    @staticmethod
    def _arimoto_blahut(transition_matrix, initial_guess, max_iters, error_tolerance):
        r"""
        Arimoto--Blahut algorithm for channel capacity. See :cite:`Cover.Thomas.06` (Sec. 10.8).
        """
        p = transition_matrix
        r = initial_guess
        last_r = np.full_like(r, fill_value=np.inf)
        iters = 0
        while iters < max_iters and np.amax(np.abs(r - last_r)) > error_tolerance:
            last_r = r
            q = r[np.newaxis].T * p
            q /= np.sum(q, axis=0)
            r = np.product(q**p, axis=1)
            r /= np.sum(r, axis=0)
            iters += 1
        return r
