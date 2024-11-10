import numpy as np

from .ScalarQuantizer import ScalarQuantizer


class UniformQuantizer(ScalarQuantizer):
    r"""
    Uniform scalar quantizer. It is a [scalar quantizer](/ref/ScalarQuantizer) in which the separation between levels is constant, $\Delta$, and the thresholds are the mid-point between adjacent levels.
    """

    def __init__(self, num_levels, input_peak=1.0, choice="mid-riser"):
        r"""
        Constructor for the class.

        Parameters:
            num_levels (int): The number of quantization levels $L$.

            input_peak (Optional[float]): The peak of the input signal $x_\mathrm{p}$. The default value is `1.0`.

            choice (Optional[str]): The choice for the uniform quantizer. Must be one of `'unsigned'` | `'mid-riser'` | `'mid-tread'`. The default value is `'mid-riser'`.

        Examples:
            >>> quantizer = komm.UniformQuantizer(num_levels=8)
            >>> quantizer.levels
            array([-0.875, -0.625, -0.375, -0.125,  0.125,  0.375,  0.625,  0.875])
            >>> quantizer.thresholds
            array([-0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75])
            >>> x = np.linspace(-0.5, 0.5, num=11)
            >>> y = quantizer(x)
            >>> np.vstack([x, y])  # doctest: +NORMALIZE_WHITESPACE
            array([[-0.5  , -0.4  , -0.3  , -0.2  , -0.1  ,  0.   ,  0.1  ,  0.2  ,  0.3  ,  0.4  ,  0.5  ],
                   [-0.375, -0.375, -0.375, -0.125, -0.125,  0.125,  0.125,  0.125,  0.375,  0.375,  0.625]])

            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_peak=1.0, choice='unsigned')
            >>> quantizer.levels
            array([0.  , 0.25, 0.5 , 0.75])
            >>> quantizer.thresholds
            array([0.125, 0.375, 0.625])

            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_peak=1.0, choice='mid-riser')
            >>> quantizer.levels
            array([-0.75, -0.25,  0.25,  0.75])
            >>> quantizer.thresholds
            array([-0.5,  0. ,  0.5])

            >>> quantizer = komm.UniformQuantizer(num_levels=4, input_peak=1.0, choice='mid-tread')
            >>> quantizer.levels
            array([-1. , -0.5,  0. ,  0.5])
            >>> quantizer.thresholds
            array([-0.75, -0.25,  0.25])
        """
        delta = (
            input_peak / num_levels
            if choice == "unsigned"
            else 2.0 * input_peak / num_levels
        )

        if choice == "unsigned":
            min_level = 0.0
            max_level = input_peak
            levels = np.linspace(min_level, max_level, num=num_levels, endpoint=False)
        elif choice == "mid-riser":
            min_level = -input_peak + (delta / 2) * (num_levels % 2 == 0)
            levels = np.linspace(
                min_level, -min_level, num=num_levels, endpoint=(num_levels % 2 == 0)
            )
        elif choice == "mid-tread":
            min_level = -input_peak + (delta / 2) * (num_levels % 2 == 1)
            levels = np.linspace(
                min_level, -min_level, num=num_levels, endpoint=(num_levels % 2 == 1)
            )
        else:
            raise ValueError(
                "parameter 'choice' must be in {'unsigned', 'mid-riser', 'mid-tread'}"
            )

        thresholds = (levels + delta / 2)[:-1]
        super().__init__(levels, thresholds)

        self._quantization_step = delta
        self._input_peak = float(input_peak)
        self._choice = choice

    @property
    def quantization_step(self):
        r"""
        The quantization step $\Delta$.
        """
        return self._quantization_step

    @property
    def input_peak(self):
        r"""
        The peak of the input signal $x_\mathrm{p}$.
        """
        return self._input_peak

    @property
    def choice(self):
        r"""
        The choice for the uniform quantizer (`'unsigned'` | `'mid-riser'` | `'mid-tread'`).
        """
        return self._choice

    def __call__(self, input_signal):
        input_signal = np.array(input_signal, dtype=float, ndmin=1)
        delta = self._quantization_step
        if self._choice in ["unsigned", "mid-tread"]:
            quantized = delta * np.floor(input_signal / delta + 0.5)
        else:  # self._choice == "mid-riser"
            quantized = delta * (np.floor(input_signal / delta) + 0.5)
        output_signal = np.clip(
            quantized, a_min=self._levels[0], a_max=self._levels[-1]
        )
        return output_signal

    def __repr__(self):
        args = "num_levels={}, input_peak={}, choice='{}'".format(
            self._num_levels, self._input_peak, self._choice
        )
        return "{}({})".format(self.__class__.__name__, args)
