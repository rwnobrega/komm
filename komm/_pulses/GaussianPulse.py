import numpy as np

from .Pulse import Pulse


class GaussianPulse(Pulse):
    """
    Gaussian pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) = \\mathrm{e}^{-\\frac{1}{2} (2 \\pi \\bar{B} t)^2}

    where the :math:`\\bar{B} = B / \\sqrt{\\ln 2}`, and :math:`B` is the half-power bandwidth of the filter.

    The Gaussian pulse is depicted below for :math:`B = 0.5`, and for :math:`B = 1`.

    .. rst-class:: centered

       |fig1| |quad| |quad| |quad| |quad| |fig2|

    .. |fig1| image:: figures/pulse_gaussian_50.png
       :alt: Gaussian pulse with half-power bandwidth of 0.5

    .. |fig2| image:: figures/pulse_gaussian_100.png
       :alt: Gaussian pulse with half-power bandwidth of 1

    .. |quad| unicode:: 0x2001
       :trim:
    """

    def __init__(self, half_power_bandwidth, length_in_symbols):
        """
        Constructor for the class. It expects the following parameters:

        :code:`half_power_bandwidth` : :obj:`float`
            The half-power bandwidth :math:`B` of the pulse.

        :code:`length_in_symbols` : :obj:`int`
            The length (span) of the truncated impulse response, in symbols.

        .. rubric:: Examples

        >>> pulse = komm.GaussianPulse(half_power_bandwidth=0.5, length_in_symbols=4)

        >>> pulse = komm.GaussianPulse(half_power_bandwidth=1.0, length_in_symbols=2)
        """
        B = self._half_power_bandwidth = float(half_power_bandwidth)
        L = self._length_in_symbols = int(length_in_symbols)
        B_bar = B / np.sqrt(np.log(2))

        def impulse_response(t):
            return np.exp(-0.5 * (2 * np.pi * B_bar * t) ** 2)

        def frequency_response(f):
            return 1 / (np.sqrt(2 * np.pi) * B_bar) * np.exp(-0.5 * (f / B_bar) ** 2)

        super().__init__(impulse_response, frequency_response, interval=(-L / 2, L / 2))

    def __repr__(self):
        args = "half_power_bandwidth={}, length_in_symbols={}".format(
            self._half_power_bandwidth, self._length_in_symbols
        )
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def half_power_bandwidth(self):
        """
        The half-power bandwidth :math:`B` of the pulse. This property is read-only.
        """
        return self._half_power_bandwidth

    @property
    def length_in_symbols(self):
        """
        The length (span) of the truncated impulse response. This property is read-only.
        """
        return self._length_in_symbols
