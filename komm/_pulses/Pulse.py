import numpy as np


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
        args = "impulse_response={}, interval={}".format(self._impulse_response, self._interval)
        return "{}({})".format(self.__class__.__name__, args)

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
