import numpy as np

__all__ = ['RectangularPulse', 'ManchesterPulse',
           'SincPulse', 'RaisedCosinePulse', 'RootRaisedCosinePulse',
           'GaussianPulse',
           'TransmitFilter', 'ReceiveFilter']


class Pulse:
    """
    General pulse.
    """
    def __init__(self, impulse_response=None, frequency_response=None, interval=None):
        """
        Constructor for the class. It expects the following parameter:

        :code:`impulse_response` : :obj:`function`
            The impulse response of the pulse.

        :code:`frequency_response` : :obj:`function`
            The frequency response of the pulse.
        """
        if impulse_response:
            self._impulse_response = np.vectorize(impulse_response)
        if frequency_response:
            self._frequency_response = np.vectorize(frequency_response)
        self._interval = interval

    def __repr__(self):
        args = 'impulse_response={}, interval={}'.format(self._impulse_response, self._interval)
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def impulse_response(self):
        """
        The impulse response of the pulse. This property is read-only.
        """
        return self._impulse_response

    @property
    def frequency_response(self):
        """
        The frequency response of the pulse. This property is read-only.
        """
        return self._frequency_response

    @property
    def interval(self):
        """
        The interval the pulse. This property is read-only.
        """
        return self._interval


class RectangularPulse(Pulse):
    """
    Rectangular pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) =
        \\begin{cases}
            1, & 0 \\leq t < w, \\\\
            0, & \\text{otherwise}.
        \\end{cases},

    where :math:`w` is the *width* of the pulse, which must satisfy :math:`0 \\leq w \\leq 1`. The rectangular pulse is depicted below for :math:`w = 1` (called the :term:`NRZ` pulse), and for :math:`w = 0.5` (called the halfway :term:`RZ` pulse).

    .. rst-class:: centered

       |fig1| |quad| |quad| |quad| |quad| |fig2|

    .. |fig1| image:: figures/pulse_rectangular_nrz.png
       :alt: Rectangular NRZ pulse

    .. |fig2| image:: figures/pulse_rectangular_rz.png
       :alt: Rectangular RZ pulse

    .. |quad| unicode:: 0x2001
       :trim:
    """
    def __init__(self, width=1.0):
        """
        Constructor for the class. It expects the following parameter:

        :code:`width` : :obj:`float`
            The width :math:`w` of the pulse. Must satisfy :math:`0 \\leq w \\leq 1`. The default value is :code:`1.0`.

        .. rubric:: Examples

        >>> pulse =  komm.RectangularPulse(width=1.0)

        >>> pulse =  komm.RectangularPulse(width=0.5)
        """
        w = self._width = float(width)

        def impulse_response(t):
            return 1.0 * (0 <= t < w)

        super().__init__(impulse_response, interval=(0.0, 1.0))

    def __repr__(self):
        args = 'width={}'.format(self._width)
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def width(self):
        """
        The width :math:`w` of the pulse. This property is read-only.
        """
        return self._width


class ManchesterPulse(Pulse):
    """
    Manchester pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) =
        \\begin{cases}
            -1, & 0 \\leq t <  1/2, \\\\
            1, & 1/2 \\leq t < 1, \\\\
            0, & \\text{otherwise},
        \\end{cases}

    The Manchester pulse is depicted below.

    .. image:: figures/pulse_manchester.png
       :alt: Manchester pulse
       :align: center
    """
    def __init__(self):
        """
        Constructor for the class. It expects no parameters.

        .. rubric:: Examples

        >>> pulse = komm.ManchesterPulse()
        """
        def impulse_response(t):
            return -1.0 * (0 <= t < 0.5) + 1.0 * (0.5 <= t < 1)

        super().__init__(impulse_response, interval=(0.0, 1.0))

    def __repr__(self):
        args = ''
        return '{}({})'.format(self.__class__.__name__, args)


class SincPulse(Pulse):
    """
    Sinc pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) = \\operatorname{sinc}(t) = \\frac{\\sin(\\pi t)}{\\pi t}.

    The sinc pulse is depicted below.

    .. image:: figures/pulse_sinc.png
       :alt: Sinc pulse
       :align: center
    """
    def __init__(self, length_in_symbols):
        """
        Constructor for the class. It expects the following parameters:

        :code:`length_in_symbols` : :obj:`int`
            The length (span) of the truncated impulse response, in symbols.

        .. rubric:: Examples

        >>> pulse = komm.SincPulse(length_in_symbols=64)
        """
        L = self._length_in_symbols = int(length_in_symbols)

        def impulse_response(t):
            return np.sinc(t)

        super().__init__(impulse_response, interval=(-L/2, L/2))

    @property
    def length_in_symbols(self):
        """
        The length (span) of the truncated impulse response. This property is read-only.
        """
        return self._length_in_symbols

    def __repr__(self):
        args = 'length_in_symbols={}'.format(self._length_in_symbols)
        return '{}({})'.format(self.__class__.__name__, args)


class RaisedCosinePulse(Pulse):
    """
    Raised cosine pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) = \\operatorname{sinc}(t) \\frac{\\cos(\\pi \\alpha t)}{1 - (2 \\alpha t)^2},

    where :math:`\\alpha` is the *rolloff factor*. The raised cosine pulse is depicted below for :math:`\\alpha = 0.25`, and for :math:`\\alpha = 0.75`.

    .. rst-class:: centered

       |fig1| |quad| |quad| |quad| |fig2|

    .. |fig1| image:: figures/pulse_raised_cosine_25.png
       :alt: Raised cosine pulse with rolloff factor 0.25

    .. |fig2| image:: figures/pulse_raised_cosine_75.png
       :alt: Raised cosine pulse with rolloff factor 0.75

    .. |quad| unicode:: 0x2001
       :trim:

    For  :math:`\\alpha = 0`, the raised cosine pulse reduces to the sinc pulse (:class:`SincPulse`).
    """
    def __init__(self, rolloff, length_in_symbols):
        """
        Constructor for the class. It expects the following parameters:

        :code:`rolloff` : :obj:`float`
            The rolloff factor :math:`\\alpha` of the pulse. Must satisfy :math:`0 \\leq \\alpha \\leq 1`.

        :code:`length_in_symbols` : :obj:`int`
            The length (span) of the truncated impulse response, in symbols.

        .. rubric:: Examples

        >>> pulse = komm.RaisedCosinePulse(rolloff=0.25, length_in_symbols=16)

        >>> pulse = komm.RaisedCosinePulse(rolloff=0.75, length_in_symbols=16)
        """
        a = self._rolloff = float(rolloff)
        L = self._length_in_symbols = int(length_in_symbols)

        def impulse_response(t):
            t += 1e-8
            return np.sinc(t) * np.cos(np.pi*a*t) / (1 - (2*a*t)**2)

        def frequency_response(f):
            f1 = (1 - a) / 2
            f2 = (1 + a) / 2
            H = 1.0 * (abs(f) < f1)
            if a > 0:
                H += (f1 < abs(f) < f2) * (0.5 + 0.5 * np.cos((np.pi * (abs(f) - f1)) / (f2 - f1)))
            return H

        super().__init__(impulse_response, frequency_response, interval=(-L/2, L/2))

    @property
    def rolloff(self):
        """
        The rolloff factor :math:`\\alpha` of the pulse. This property is read-only.
        """
        return self._rolloff

    @property
    def length_in_symbols(self):
        """
        The length (span) of the truncated impulse response. This property is read-only.
        """
        return self._length_in_symbols

    def __repr__(self):
        args = 'rolloff={}, length_in_symbols={}'.format(self._rolloff, self._length_in_symbols)
        return '{}({})'.format(self.__class__.__name__, args)


class RootRaisedCosinePulse(Pulse):
    """
    Root raised cosine pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) = \\frac{\\sin[\\pi (1 - \\alpha) t] + 4 \\alpha t \\cos[\\pi (1 + \\alpha) t]}{\\pi t [1 - (4 \\alpha t)^2]},

    where :math:`\\alpha` is the *rolloff factor*. The root raised cosine pulse is depicted below for :math:`\\alpha = 0.25`, and for :math:`\\alpha = 0.75`.

    .. rst-class:: centered

       |fig1| |quad| |quad| |quad| |fig2|

    .. |fig1| image:: figures/pulse_root_raised_cosine_25.png
       :alt: Root raised cosine pulse with rolloff factor 0.25

    .. |fig2| image:: figures/pulse_root_raised_cosine_75.png
       :alt: Root raised cosine pulse with rolloff factor 0.75

    .. |quad| unicode:: 0x2001
       :trim:
    """
    def __init__(self, rolloff, length_in_symbols):
        """
        Constructor for the class. It expects the following parameters:

        :code:`rolloff` : :obj:`float`
            The rolloff factor :math:`\\alpha` of the pulse. Must satisfy :math:`0 \\leq \\alpha \\leq 1`.

        :code:`length_in_symbols` : :obj:`int`
            The length (span) of the truncated impulse response, in symbols.

        .. rubric:: Examples

        >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.25, length_in_symbols=16)

        >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.75, length_in_symbols=16)
        """
        a = self._rolloff = float(rolloff)
        L = self._length_in_symbols = int(length_in_symbols)

        def impulse_response(t):
            t += 1e-8
            return (np.sin(np.pi*(1 - a)*t) + 4*a*t * np.cos(np.pi*(1 + a)*t)) / (np.pi*t*(1 - (4*a*t)**2))

        super().__init__(impulse_response, interval=(-L/2, L/2))

    @property
    def rolloff(self):
        """
        The rolloff factor :math:`\\alpha` of the pulse. This property is read-only.
        """
        return self._rolloff

    @property
    def length_in_symbols(self):
        """
        The length (span) of the truncated impulse response. This property is read-only.
        """
        return self._length_in_symbols

    def __repr__(self):
        args = 'rolloff={}, length_in_symbols={}'.format(self._rolloff, self._length_in_symbols)
        return '{}({})'.format(self.__class__.__name__, args)


class GaussianPulse(Pulse):
    """
    Gaussian pulse. It is a formatting pulse (:class:`FormattingPulse`) with impulse response given by

    .. math::

        h(t) = \\mathrm{e}^{-\\frac{1}{2} (2 \\pi \\bar{B} t)^2}

    where the :math:`\\bar{B} = B / \\sqrt{\\ln 2}`, and :math:`B` is the half-power bandwidth of the filter.

    The gaussian pulse is depicted below for :math:`B = 0.5`, and for :math:`B = 1`.

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

        >>> pulse =  komm.GaussianPulse(half_power_bandwidth=0.5, length_in_symbols=4)

        >>> pulse =  komm.GaussianPulse(half_power_bandwidth=1.0, length_in_symbols=2)
        """
        B = self._half_power_bandwidth = float(half_power_bandwidth)
        L = self._length_in_symbols = int(length_in_symbols)
        B_bar = B / np.sqrt(np.log(2))

        def impulse_response(t):
            return np.exp(-0.5 * (2*np.pi*B_bar*t)**2)

        def frequency_response(f):
            return 1 / (np.sqrt(2*np.pi) * B_bar) * np.exp(-0.5 * (f / B_bar)**2)

        super().__init__(impulse_response, frequency_response, interval=(-L/2, L/2))

    def __repr__(self):
        args = 'half_power_bandwidth={}, length_in_symbols={}'.format(self._half_power_bandwidth, self._length_in_symbols)
        return '{}({})'.format(self.__class__.__name__, args)

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


class TransmitFilter:
    """
    Transmit filter.
    """
    def __init__(self, pulse, samples_per_symbol):
        """
        Constructor for the class. It expects the following parameters:

        :code:`pulse` : :class:`komm.Pulse`
            The pulse filter.

        :code:`samples_per_symbol` : :obj:`int`
            The number of samples (of the output) per symbol (of the input).
        """
        self.Pulse = pulse
        self._samples_per_symbol = int(samples_per_symbol)

    @property
    def pulse(self):
        """
        The pulse filter. This property is read-only.
        """
        return self.Pulse

    @property
    def samples_per_symbol(self):
        """
        The number of samples per symbol. This property is read-only.
        """
        return self._samples_per_symbol

    def __call__(self, inp):
        """
        Formats a sequence of symbols.

        **Input:**

        :code:`inp` : 1D-array of :obj:`float`
            The input signal, containing symbols of a modulation.

        **Output:**

        :code:`outp` : 1D-array of :obj:`float`
            The output signal, formatted.
        """
        sps = self._samples_per_symbol
        t0, t1 = self.Pulse.interval
        t = np.arange(t0, t1, step=1/sps)
        taps = (np.vectorize(self.Pulse.impulse_response))(t)
        inp_interp = np.zeros((len(inp) - 1) * sps + 1, dtype=np.float)
        inp_interp[::sps] = inp
        outp = np.convolve(taps, inp_interp)
        return outp


class ReceiveFilter:
    """
    Receive filter [Not implemented yet].
    """
    pass
