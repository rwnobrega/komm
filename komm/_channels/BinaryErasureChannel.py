import numpy as np

from .DiscreteMemorylessChannel import DiscreteMemorylessChannel


class BinaryErasureChannel(DiscreteMemorylessChannel):
    r"""
    Binary erasure channel (BEC). It is a discrete memoryless channel (:obj:`DiscreteMemorylessChannel`) with input alphabet :math:`\mathcal{X} = \{ 0, 1 \}`, output alphabet :math:`\mathcal{Y} = \{ 0, 1, 2 \}`, and transition probability matrix given by

    .. math::

        p_{Y \mid X} =
        \begin{bmatrix}
            1 - \epsilon & 0 & \epsilon \\
            0 & 1 - \epsilon & \epsilon
        \end{bmatrix},

    where the parameter :math:`\epsilon` is called the *erasure probability* of the channel. See :cite:`Cover.Thomas.06` (Sec. 7.1.5).

    To invoke the channel, call the object giving the input signal as parameter (see example in the constructor below).
    """

    def __init__(self, erasure_probability=0.0):
        r"""
        Constructor for the class.

        Parameters:

            erasure_probability (:obj:`float`, optional): The channel erasure probability :math:`\epsilon`. Must satisfy :math:`0 \leq \epsilon \leq 1`. Default value is :code:`0.0`, which corresponds to a noiseless channel.

        Examples:

            >>> bec = komm.BinaryErasureChannel(0.1)
            >>> x = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
            >>> y = bec(x); y  #doctest: +SKIP
            array([1, 1, 1, 2, 0, 0, 1, 0, 1, 0])
        """
        self.erasure_probability = erasure_probability

    @property
    def erasure_probability(self):
        r"""
        The erasure probability :math:`\epsilon` of the channel. This is a read-and-write property.
        """
        return self._erasure_probability

    @erasure_probability.setter
    def erasure_probability(self, value):
        self._erasure_probability = e = float(value)
        self.transition_matrix = [[1 - e, 0, e], [0, 1 - e, e]]

    def capacity(self):
        r"""
        Returns the channel capacity :math:`C`. It is given by :math:`C = 1 - \epsilon`. See :cite:`Cover.Thomas.06` (Sec. 7.1.5).

        Examples:

            >>> bec = komm.BinaryErasureChannel(0.25)
            >>> bec.capacity()
            0.75
        """
        return 1.0 - self._erasure_probability

    def __call__(self, input_sequence):
        erasure_pattern = np.random.rand(np.size(input_sequence)) < self._erasure_probability
        output_sequence = np.copy(input_sequence)
        output_sequence[erasure_pattern] = 2
        return output_sequence

    def __repr__(self):
        args = "erasure_probability={}".format(self._erasure_probability)
        return "{}({})".format(self.__class__.__name__, args)
