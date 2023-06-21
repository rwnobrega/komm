import numpy as np


class FormattingPulse:
    r"""
    General formatting pulse.
    """

    def __init__(self, impulse_response=None, frequency_response=None, interval=None):
        r"""
        Constructor for the class.

        Parameters:

            impulse_response (function): The impulse response of the pulse.

            frequency_response (function): The frequency response of the pulse.
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
        r"""
        The impulse response of the pulse.
        """
        return self._impulse_response

    @property
    def frequency_response(self):
        r"""
        The frequency response of the pulse.
        """
        return self._frequency_response

    @property
    def interval(self):
        r"""
        The interval the pulse.
        """
        return self._interval
