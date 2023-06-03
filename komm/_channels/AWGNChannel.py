import numpy as np


class AWGNChannel:
    r"""
    Additive white Gaussian noise (AWGN) channel. It is defined by

    .. math::

        Y_n = X_n + Z_n,

    where :math:`X_n` is the channel *input signal*, :math:`Y_n` is the channel *output signal*, and :math:`Z_n` is the *noise*, which is :term:`i.i.d.` according to a Gaussian distribution with zero mean. The channel *signal-to-noise ratio* is calculated by

    .. math::

        \mathrm{SNR} = \frac{P}{N},

    where :math:`P = \mathrm{E}[X^2_n]` is the average power of the input signal, and :math:`N = \mathrm{E}[Z^2_n]` is the average power (and variance) of the noise.

    References: :cite:`Cover.Thomas.06` (Ch. 9)

    To invoke the channel, call the object giving the input signal as parameter (see example below).
    """

    def __init__(self, snr=np.inf, signal_power=1.0):
        r"""Constructor for the class. It expects the following parameters:

        :code:`snr` : :obj:`float`, optional
            The channel signal-to-noise ratio :math:`\mathrm{SNR}` (linear, not decibel). The default value is :code:`np.inf`.

        :code:`signal_power` : :obj:`float` or :obj:`str`, optional
            The input signal power :math:`P`.  If equal to the string :code:`'measured'`, then every time the channel is invoked the input signal power will be computed from the input itself (i.e., its squared Euclidean norm). The default value is :code:`1.0`.

        .. rubric:: Examples

        >>> awgn = komm.AWGNChannel(snr=100.0, signal_power=1.0)
        >>> x = [1.0, 3.0, -3.0, -1.0, -1.0, 1.0, 3.0, 1.0, -1.0, 3.0]
        >>> y = awgn(x); y  #doctest:+SKIP
        array([ 1.10051445,  3.01308154, -2.97997111, -1.1229903 , -0.90890299,
                1.12650432,  2.88952462,  0.99352172, -1.2072787 ,  3.27131731])
        """
        self.snr = snr
        self.signal_power = signal_power

    @property
    def snr(self):
        r"""
        The signal-to-noise ratio :math:`\mathrm{SNR}` (linear, not decibel) of the channel. This is a read-and-write property.
        """
        return self._snr

    @snr.setter
    def snr(self, value):
        self._snr = float(value)

    @property
    def signal_power(self):
        r"""
        The input signal power :math:`P`. This is a read-and-write property.
        """
        return self._signal_power

    @signal_power.setter
    def signal_power(self, value):
        self._signal_power = float(value)

    def capacity(self):
        r"""
        Returns the channel capacity :math:`C`. It is given by :math:`C = \frac{1}{2}\log_2(1 + \mathrm{SNR})`, in bits per dimension.

        .. rubric:: Examples

        >>> awgn = komm.AWGNChannel(snr=63.0)
        >>> awgn.capacity()
        3.0
        """
        return 0.5 * np.log1p(self._snr) / np.log(2.0)

    def __call__(self, input_signal):
        input_signal = np.array(input_signal)
        size = input_signal.size

        if self._signal_power == "measured":
            signal_power = np.linalg.norm(input_signal) ** 2 / size
        else:
            signal_power = self._signal_power

        noise_power = signal_power / self._snr

        if input_signal.dtype == complex:
            noise = np.sqrt(noise_power / 2) * (np.random.normal(size=size) + 1j * np.random.normal(size=size))
        else:
            noise = np.sqrt(noise_power) * np.random.normal(size=size)

        return input_signal + noise

    def __repr__(self):
        args = "snr={}, signal_power={}".format(self._snr, self._signal_power)
        return "{}({})".format(self.__class__.__name__, args)
