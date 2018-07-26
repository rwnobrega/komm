import numpy as np

__all__ = ['ScalarQuantizer', 'UniformQuantizer']


class ScalarQuantizer:
    """
    General scalar quantizer. It is defined by a list of *levels*, :math:`v_0, v_1, \\ldots, v_{L-1}`, and a list of *thresholds*, :math:`t_0, t_1, \\ldots, t_L`, satisfying

    .. math::

       -\\infty = t_0 < v_0 < t_1 < v_1 < \\cdots < t_{L - 1} < v_{L - 1} < t_L = +\\infty.

    Given an input :math:`x \\in \\mathbb{R}`, the output of the quantizer is given by :math:`y = v_i` if and only if :math:`t_i \leq x < t_{i+1}`, where :math:`i \\in [0:L)`.
    """
    def __init__(self, levels, thresholds):
        """
        Constructor for the class. It expects the following parameters:

        :code:`levels` : 1D array of :obj:`float`
            The quantizer levels :math:`v_0, v_1, \\ldots, v_{L-1}`. It should be a list floats of length :math:`L`.

        :code:`thresholds` : 1D array of :obj:`float`
            The finite quantizer thresholds :math:`t_1, t_2, \\ldots, t_{L-1}`. It should be a list of floats of length :math:`L - 1`.

        Moreover, they must satisfy :math:`v_0 < t_1 < v_1 < \\cdots < t_{L - 1} < v_{L - 1}`.

        .. rubric:: Examples

        >>> quantizer = komm.ScalarQuantizer(levels=[-1.0, 0.0, 1.0], thresholds=[-0.5, 0.8])
        >>> x = np.linspace(-2.5, 2.5, num=11)
        >>> y = quantizer(x)
        >>> np.vstack([x, y])
        array([[-2.5, -2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ,  2.5],
               [-1. , -1. , -1. , -1. , -1. ,  0. ,  0. ,  1. ,  1. ,  1. ,  1. ]])
        """
        self._levels = np.array(levels, dtype=np.float)
        self._thresholds = np.array(thresholds, dtype=np.float)
        self._num_levels = self._levels.size

        if self._thresholds.size != self._num_levels - 1:
            raise ValueError("The length of 'thresholds' must be 'num_levels - 1'")

        interleaved = np.empty(2*self._num_levels - 1, dtype=np.float)
        interleaved[0::2] = self._levels
        interleaved[1::2] = self._thresholds

        if not np.array_equal(np.unique(interleaved), interleaved):
            raise ValueError("Invalid values for 'levels' and 'thresholds'")

    @property
    def levels(self):
        """
        The quantizer levels, :math:`v_0, v_1, \\ldots, v_{L-1}`.
        """
        return self._levels

    @property
    def thresholds(self):
        """
        The finite quantizer thresholds, :math:`t_1, t_2, \\ldots, t_{L-1}`.
        """
        return self._thresholds

    @property
    def num_levels(self):
        """
        The number of quantization levels, :math:`L`.
        """
        return self._num_levels

    def __call__(self, input_signal):
        input_signal_tile = np.tile(input_signal, reps=(self._thresholds.size, 1)).transpose()
        output_signal = self._levels[np.sum(input_signal_tile > self._thresholds, axis=1)]
        return output_signal

    def __repr__(self):
        args = 'levels={}, thresholds={}'.format(self.levels.tolist(), self.thresholds.tolist())
        return '{}({})'.format(self.__class__.__name__, args)


class UniformQuantizer(ScalarQuantizer):
    """
    Uniform scalar quantizer. It is a scalar quantizer (:obj:`ScalarQuantizer`) in which the separation between the levels is a constant, and the thresholds are the mid-point between adjacent levels.
    """
    def __init__(self, num_levels):
        """
        Constructor for the class. It expects the following parameters:
        """
        pass
