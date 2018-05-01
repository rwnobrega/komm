"""
Pulses
======

Pulses...

Base class
----------

.. autosummary::
    :nosignatures:
    :toctree: stubs/

    Pulse

Rectangular pulses
------------------

.. autosummary::
    :nosignatures:
    :toctree: stubs/

    RectangularNRZPulse
    RectangularRZPulse
    ManchesterPulse

Nyquist pulses
--------------

.. autosummary::
    :nosignatures:
    :toctree: stubs/

    SincPulse
    RaisedCosinePulse
    RootRaisedCosinePulse

Other pulses
------------

.. autosummary::
    :nosignatures:
    :toctree: stubs/

    GaussianPulse
"""

import numpy as np

__all__ = ['Pulse',
           'RectangularNRZPulse', 'RectangularRZPulse',
           'SincPulse', 'RaisedCosinePulse', 'RootRaisedCosinePulse',
           'GaussianPulse']


class Pulse:
    def __init__(self, impulse_response, samples_per_symbol):
        self._impulse_response = np.array(impulse_response, dtype=np.float)
        self._samples_per_symbol = samples_per_symbol

    @property
    def impulse_response(self):
        return self._impulse_response

    @property
    def samples_per_symbol(self):
        return self._samples_per_symbol

    def filter(self, signal):
        sps = self._samples_per_symbol
        signal_interp = np.zeros(len(signal) * sps, dtype=np.float)
        signal_interp[::sps] = signal
        filtered = np.convolve(self._impulse_response, signal_interp)
        return filtered

    def __repr__(self):
        return f'{self.__class__.__name__}(impulse_response={self._impulse_response.tolist()}, ' \
               f'samples_per_symbol={self._samples_per_symbol})'


class RectangularNRZPulse(Pulse):
    """
    Rectangular non-return to zero (NRZ) pulse.
    """
    def __init__(self, samples_per_symbol):
        super().__init__(np.ones(samples_per_symbol, dtype=np.float), samples_per_symbol)

    def __repr__(self):
        return f'{self.__class__.__name__}(samples_per_symbol={self._samples_per_symbol})'


class RectangularRZPulse(Pulse):
    """
    Rectangular return to zero (RZ) pulse.
    """
    def __init__(self, samples_per_symbol):
        if samples_per_symbol % 2 == 0:
            middle = np.array([])
        else:
            middle = np.array([0.5])
        impulse_response = np.concatenate((np.ones(samples_per_symbol // 2, dtype=np.float),
                                           middle,
                                           np.zeros(samples_per_symbol // 2, dtype=np.float)))
        super().__init__(impulse_response, samples_per_symbol)

    def __repr__(self):
        return f'{self.__class__.__name__}(samples_per_symbol={self._samples_per_symbol})'



class ManchesterPulse(Pulse):
    """
    Manchester pulse.
    """
    def __init__(self, samples_per_symbol):
        if samples_per_symbol % 2 == 0:
            middle = np.array([])
        else:
            middle = np.array([0.0])
        impulse_response = np.concatenate((-np.ones(samples_per_symbol // 2, dtype=np.float),
                                           middle,
                                           np.ones(samples_per_symbol // 2, dtype=np.float)))
        super().__init__(impulse_response, samples_per_symbol)

    def __repr__(self):
        return f'{self.__class__.__name__}(samples_per_symbol={self._samples_per_symbol})'


class SincPulse(Pulse):
    def __init__(self, samples_per_symbol, length_in_symbols):
        L = samples_per_symbol * length_in_symbols // 2
        t = np.arange(-L, L) / samples_per_symbol
        impulse_response = np.sinc(t)
        super().__init__(impulse_response, samples_per_symbol)
        self._length_in_symbols = length_in_symbols

    def __repr__(self):
        return f'{self.__class__.__name__}(samples_per_symbol={self._samples_per_symbol}, ' \
               f'length_in_symbols={self._length_in_symbols})'


class RaisedCosinePulse(Pulse):
    def __init__(self, rolloff, samples_per_symbol, length_in_symbols):
        L = samples_per_symbol * length_in_symbols // 2
        epsilon = np.finfo(np.float).eps
        t = np.arange(-L, L) / samples_per_symbol + epsilon
        impulse_response = np.sin(np.pi * t) / (np.pi * t) * \
                           np.cos(np.pi * rolloff * t) / (1 - (2 * rolloff * t)**2)
        super().__init__(impulse_response, samples_per_symbol)
        self._length_in_symbols = length_in_symbols
        self._rolloff = rolloff

    def __repr__(self):
        return f'{self.__class__.__name__}(rolloff={self._rolloff}, ' \
               f'samples_per_symbol={self._samples_per_symbol}, ' \
               f'length_in_symbols={self._length_in_symbols})'


class RootRaisedCosinePulse(Pulse):
    def __init__(self, rolloff, samples_per_symbol, length_in_symbols):
        L = samples_per_symbol * length_in_symbols // 2
        epsilon = np.finfo(np.float).eps
        t = np.arange(-L, L) / samples_per_symbol + epsilon
        impulse_response = (np.sin(np.pi * (1 - rolloff) * t) +
                            4 * rolloff * t * np.cos(np.pi * (1 + rolloff) * t)) / \
                           (np.pi * t * (1 - (4 * rolloff * t)**2))
        super().__init__(impulse_response, samples_per_symbol)
        self._length_in_symbols = length_in_symbols
        self._rolloff = rolloff

    def __repr__(self):
        return f'{self.__class__.__name__}(rolloff={self._rolloff}, ' \
               f'samples_per_symbol={self._samples_per_symbol}, ' \
               f'length_in_symbols={self._length_in_symbols})'


class GaussianPulse(Pulse):
    pass
