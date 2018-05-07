import numpy as np

__all__ = ['FormattingPulse',
           'RectangularNRZPulse', 'RectangularRZPulse', 'ManchesterPulse',
           'SincPulse', 'RaisedCosinePulse', 'RootRaisedCosinePulse',
           'GaussianPulse']


class FormattingPulse:
    """
    General formatting pulse.
    """
    def __init__(self, impulse_response, samples_per_symbol):
        """
        Constructor for the class. It expects the following parameters:

        :code:`impulse_response` : 1D array of :obj:`float`
            The filter finite impulse response.

        :code:`samples_per_symbol` : :obj:`int`
            The number of samples (of the impulse response) per symbol (of the modulation). 
        """
        self._impulse_response = np.array(impulse_response, dtype=np.float)
        self._samples_per_symbol = samples_per_symbol

    @property
    def impulse_response(self):
        """
        Impulse response of the formatting pulse. This property is read-only.
        """
        return self._impulse_response

    @property
    def samples_per_symbol(self):
        """
        Samples per symbol. This property is read-only.
        """
        return self._samples_per_symbol

    def format(self, symbols):
        """
        Formats a sequence of symbols.

        **Input:**

        :code:`symbols` : 1D array of :obj:`float`
            The input signal, containing symbols of the modulation.

        **Output:**

        :code:`formatted` : 1D array of :obj:`float`
            The output, formatted signal.
        """
        sps = self._samples_per_symbol
        signal_interp = np.zeros(len(signal) * sps, dtype=np.float)
        signal_interp[::sps] = symbols
        formatted = np.convolve(self._impulse_response, signal_interp)
        return filtered

    def __repr__(self):
        args = 'impulse_response={}, samples_per_symbol={}'.format(self._impulse_response.tolist(), self._samples_per_symbol)
        return '{}({})'.format(self.__class__.__name__, args)


class RectangularNRZPulse(FormattingPulse):
    """
    Rectangular non-return to zero (NRZ) pulse. Its impulse response is given by

    .. math::

        h(t) = 
        \\begin{cases}
            1, & 0 \\leq t < 1, \\\\
            0, & \\text{otherwise}.
        \\end{cases}
    """
    def __init__(self, samples_per_symbol):
        """
        Constructor for the class. It expects the following parameter:

        :code:`samples_per_symbol` : :obj:`int`
            The number of samples per symbol. 
        """
        super().__init__(np.ones(samples_per_symbol, dtype=np.float), samples_per_symbol)

    def __repr__(self):
        args = 'samples_per_symbol={}'.format(self._samples_per_symbol)
        return '{}({})'.format(self.__class__.__name__, args)


class RectangularRZPulse(FormattingPulse):
    """
    Rectangular return to zero (RZ) pulse. Its impulse response is given by

    .. math::

        h(t) = 
        \\begin{cases}
            1, & 0 \\leq t < 1/2, \\\\
            0, & \\text{otherwise},
        \\end{cases}
    """
    def __init__(self, samples_per_symbol):
        """
        Constructor for the class. It expects the following parameter:

        :code:`samples_per_symbol` : :obj:`int`
            The number of samples per symbol. 
        """
        if samples_per_symbol % 2 == 0:
            middle = np.array([])
        else:
            middle = np.array([0.5])
        impulse_response = np.concatenate((np.ones(samples_per_symbol // 2, dtype=np.float),
                                           middle,
                                           np.zeros(samples_per_symbol // 2, dtype=np.float)))
        super().__init__(impulse_response, samples_per_symbol)

    def __repr__(self):
        args = 'samples_per_symbol={}'.format(self._samples_per_symbol)
        return '{}({})'.format(self.__class__.__name__, args)


class ManchesterPulse(FormattingPulse):
    """
    Manchester pulse. Its impulse response is given by

    .. math::

        h(t) = 
        \\begin{cases}
            -1, & 0 \\leq t <  1/2, \\\\
            1, & 1/2 \\leq t < 1, \\\\
            0, & \\text{otherwise},
        \\end{cases}
    """
    def __init__(self, samples_per_symbol):
        """
        Constructor for the class. It expects the following parameter:

        :code:`samples_per_symbol` : :obj:`int`
            The number of samples per symbol. 
        """
        if samples_per_symbol % 2 == 0:
            middle = np.array([])
        else:
            middle = np.array([0.0])
        impulse_response = np.concatenate((-np.ones(samples_per_symbol // 2, dtype=np.float),
                                           middle,
                                           np.ones(samples_per_symbol // 2, dtype=np.float)))
        super().__init__(impulse_response, samples_per_symbol)

    def __repr__(self):
        args = 'samples_per_symbol={}'.format(self._samples_per_symbol)
        return '{}({})'.format(self.__class__.__name__, args)


class SincPulse(FormattingPulse):
    """
    Sinc pulse. Its impulse response is given by

    .. math::

        h(t) = \\operatorname{sinc}(t) = \\frac{\\sin(t)}{t}.

    """
    def __init__(self, samples_per_symbol, length_in_symbols):
        """
        Constructor for the class. It expects the following parameters:

        :code:`samples_per_symbol` : :obj:`int`
            The number of samples per symbol.

        :code:`length_in_symbols` : :obj:`int`
            The length (span) of the truncated impulse response, in symbols.
        """
        L = samples_per_symbol * length_in_symbols // 2
        t = np.arange(-L, L) / samples_per_symbol
        impulse_response = np.sinc(t)
        super().__init__(impulse_response, samples_per_symbol)
        self._length_in_symbols = length_in_symbols

    @property
    def length_in_symbols(self):
        """
        Length (span) of the truncated impulse response.
        """
        return self._length_in_symbols

    def __repr__(self):
        args = 'samples_per_symbol={}, length_in_symbols={}'.format(self._samples_per_symbol, self._length_in_symbols)
        return '{}({})'.format(self.__class__.__name__, args)


class RaisedCosinePulse(FormattingPulse):
    """
    Raised cosine pulse. Its impulse response is given by

    .. math::

        h(t) = \\operatorname{sinc}(t) \\frac{\\cos(\\pi \\alpha t)}{1 - (2 \\alpha t)^2},

    where :math:`\\alpha` is the rolloff factor.
    """
    def __init__(self, rolloff, samples_per_symbol, length_in_symbols):
        """
        Constructor for the class. It expects the following parameters:

        :code:`rolloff` : :obj:`float`
            The rolloff factor :math:`\\alpha` of the pulse. Must satisfy :math:`0 \\leq \\alpha \\leq 1`.
        
        :code:`samples_per_symbol` : :obj:`int`
            The number of samples per symbol.

        :code:`length_in_symbols` : :obj:`int`
            The length (span) of the truncated impulse response, in symbols.
        """
        L = samples_per_symbol * length_in_symbols // 2
        epsilon = np.finfo(np.float).eps
        t = np.arange(-L, L) / samples_per_symbol + epsilon
        impulse_response = np.sinc(t) * np.cos(np.pi * rolloff * t) / (1 - (2 * rolloff * t)**2)
        super().__init__(impulse_response, samples_per_symbol)
        self._length_in_symbols = length_in_symbols
        self._rolloff = rolloff

    @property
    def length_in_symbols(self):
        """
        Length (span) of the truncated impulse response.
        """
        return self._length_in_symbols
    
    @property
    def rolloff_factor(self):
        """
        Rolloff factor.
        """
        return self._rolloff_factor

    def __repr__(self):
        args = 'rolloff={}, samples_per_symbol={}, length_in_symbols={}'.format(self._rolloff, self._samples_per_symbol, self._length_in_symbols)
        return '{}({})'.format(self.__class__.__name__, args)


class RootRaisedCosinePulse(FormattingPulse):
    """
    Root raised cosine pulse. Its impulse response is given by

    .. math::

        h(t) = \\frac{\\sin[\\pi (1 - \\alpha) t] + 4 \\alpha t \\cos[\\pi (1 + \\alpha) t]}{\\pi t [1 - (4 \\alpha t)^2]},

    where :math:`\\alpha` is the rolloff factor.
    """
    def __init__(self, rolloff, samples_per_symbol, length_in_symbols):
        """
        Constructor for the class. It expects the following parameters:

        :code:`rolloff` : :obj:`float`
            The rolloff factor :math:`\\alpha` of the pulse. Must satisfy :math:`0 \\leq \\alpha \\leq 1`.
        
        :code:`samples_per_symbol` : :obj:`int`
            The number of samples per symbol.

        :code:`length_in_symbols` : :obj:`int`
            The length (span) of the truncated impulse response, in symbols.
        """
        L = samples_per_symbol * length_in_symbols // 2
        epsilon = np.finfo(np.float).eps
        t = np.arange(-L, L) / samples_per_symbol + epsilon
        impulse_response = (np.sin(np.pi * (1 - rolloff) * t) +
                            4 * rolloff * t * np.cos(np.pi * (1 + rolloff) * t)) / \
                           (np.pi * t * (1 - (4 * rolloff * t)**2))
        super().__init__(impulse_response, samples_per_symbol)
        self._length_in_symbols = length_in_symbols
        self._rolloff = rolloff
    
    @property
    def length_in_symbols(self):
        """
        Length (span) of the truncated impulse response.
        """
        return self._length_in_symbols
    
    @property
    def rolloff_factor(self):
        """
        Rolloff factor.
        """
        return self._rolloff_factor

    def __repr__(self):
        args = 'rolloff={}, samples_per_symbol={}, length_in_symbols={}'.format(self._rolloff, self._samples_per_symbol, self._length_in_symbols)
        return '{}({})'.format(self.__class__.__name__, args)


class GaussianPulse(FormattingPulse):
    pass
