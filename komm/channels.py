import numpy as np

from .util import \
    entropy, mutual_information

__all__ = ['AWGNChannel',
           'DiscreteMemorylessChannel',
           'BinarySymmetricChannel', 'BinaryErasureChannel']


class AWGNChannel:
    """
    Additive white gaussian noise (AWGN) channel.

    The *additive white gaussian noise* (*AWGN*) channel :cite:`Cover.Thomas.06` (Ch. 9) is defined by

    .. math::

        Y_n = X_n + Z_n,

    where :math:`X_n` is the channel *input signal*, :math:`Y_n` is the channel *output signal*, and :math:`Z_n` is the *noise*, which is :term:`i.i.d.` according to a gaussian distribution with zero mean. The channel *signal-to-noise ratio* (*SNR*) is calculated by

    .. math::

        \\mathrm{SNR} = \\frac{P}{N},

    where :math:`P = \\mathrm{E}[X^2_n]` is the average power of the input signal, and :math:`N = \\mathrm{E}[Z^2_n]` is the average power (and variance) of the noise.

    The constructor takes the channel SNR and the input signal power as parameters.

    To invoke the channel for a given input, just

    Parameters
    ==========
    snr : `float`, optional
        The channel signal-to-noise ratio :math:`\\mathrm{SNR}` (linear, not decibel). Default value is :code:`np.inf`.
    signal_power : `float` or `str`, optional
        The input signal power :math:`P`.  If set to the string :code:`'measured'`, every time the channel is invoked the input signal power will be computed from the input (i.e., its squared Euclidean norm). Default value is :code:`1.0`.

    Examples
    ========
    >>> awgn = komm.AWGNChannel(snr=100.0, signal_power=1.0)
    >>> x = np.random.choice([-3.0, -1.0, 1.0, 3.0], size=10); x
    array([ 1.,  3., -3., -1., -1.,  1.,  3.,  1., -1.,  3.])
    >>> y = awgn(x); y
    array([ 0.98966376,  2.99349135, -3.05371748, -0.71632748, -1.06163275,
            0.75899613,  2.86905731,  1.16039474, -1.02437047,  2.91492338])
    """
    def __init__(self, snr=np.inf, signal_power=1.0):
        self._snr = snr
        self._signal_power = signal_power

    @property
    def snr(self):
        """
        The channel signal-to-noise ratio :math:`\\mathrm{SNR}` (linear, not decibel).
        """
        return self._snr

    @snr.setter
    def snr(self, value):
        self._snr = value

    @property
    def signal_power(self):
        """
        The input signal power :math:`P`.
        """
        return self._signal_power

    @signal_power.setter
    def signal_power(self, value):
        self._signal_power = value

    def capacity(self):
        """
        Returns the channel capacity, given by :math:`C = \\frac{1}{2}\\log_2(1 + \\mathrm{SNR})`, in bits per dimension.

        Examples
        ========
        >>> awgn = komm.AWGNChannel(snr=63.0, signal_power=1.0)
        >>> awgn.capacity()
        3.0
        """
        return 0.5 * np.log2(1 + self._snr)

    def __call__(self, input_signal):
        input_signal = np.array(input_signal)
        size = input_signal.size

        if self._signal_power == 'measured':
            signal_power = np.linalg.norm(input_signal)**2 / size
        else:
            signal_power = self._signal_power

        noise_power = signal_power / self.snr

        if input_signal.dtype == np.complex:
            noise = np.sqrt(noise_power / 2) * (np.random.normal(size=size) + 1j * np.random.normal(size=size))
        else:
            noise = np.sqrt(noise_power) * np.random.normal(size=size)

        return input_signal + noise

    def __repr__(self):
        args = 'snr={}, signal_power={}'.format(self.snr, self.signal_power)
        return '{}({})'.format(self.__class__.__name__, args)


class DiscreteMemorylessChannel:
    """
    Discrete memoryless channel (DMC).

    A *discrete memoryless channel* (*DMC*) :cite:`Cover.Thomas.06` (Ch. 7) is defined by an *input alphabet* :math:`\\mathcal{X}`, an *output alphabet* :math:`\\mathcal{Y}`, and a *transition matrix* :math:`p_{Y \mid X}`.

    The constructor expects the transition matrix, of size :math:`|\\mathcal{X}|`-by-:math:`|\\mathcal{Y}|`, as its sole parameter. The input and output alphabets are always taken as :math:`\\mathcal{X} = \\{ 0, 1, \ldots, |\\mathcal{X}| - 1 \\}` and :math:`\\mathcal{Y} = \\{ 0, 1, \ldots, |\\mathcal{Y}| - 1 \\}`, respectively.

    Parameters
    ==========
    snr : `float`
        The channel signal-to-noise ratio :math:`\\mathrm{SNR}`. Default value is :code:`1.0`.

    Examples
    ========
    >>> dmc = komm.DiscreteMemorylessChannel([[0.9, 0.05, 0.05], [0.0, 0.5, 0.5]])
    >>> x = np.random.randint(2, size=10); x
    array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
    >>> y = dmc(x); y
    array([0, 2, 0, 2, 1, 1, 0, 0, 0, 2])
    """
    def __init__(self, transition_matrix):
        self._transition_matrix = np.array(transition_matrix, dtype=np.float)
        self._input_cardinality, self._output_cardinality = self._transition_matrix.shape
        self._arimoto_blahut_kwargs = {'max_iters': 1000, 'error_tolerance': 1e-12}

    @property
    def transition_matrix(self):
        """
        The channel transition matrix :math:`p_{Y \mid X}`.
        """
        return self._transition_matrix

    @property
    def input_cardinality(self):
        """
        The channel input cardinality :math:`|\\mathcal{X}|`.
        """
        return self._input_cardinality

    @property
    def output_cardinality(self):
        """
        The channel output cardinality :math:`|\\mathcal{Y}|`.
        """
        return self._output_cardinality

    def capacity(self):
        """
        Returns the channel capacity, calculated by the Arimoto--Blahut algorithm :cite:`Cover.Thomas.06` (Sec. 10.8).

        Examples
        ========
        >>> dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2], [0.5, 0.05, 0.45]])
        >>> dmc.capacity()
        0.1616318609548566
        """
        initial_guess = np.ones(self._input_cardinality, dtype=np.float) / self._input_cardinality
        optimal_input_pmf = arimoto_blahut(self._transition_matrix, initial_guess, **self._arimoto_blahut_kwargs)
        return mutual_information(optimal_input_pmf, self._transition_matrix)

    def __call__(self, input_sequence):
        return discrete_memoryless_channel(input_sequence, self._output_cardinality, self._transition_matrix)

    def __repr__(self):
        args = 'transition_matrix={}'.format(self._transition_matrix.tolist())
        return '{}({})'.format(self.__class__.__name__, args)


class BinarySymmetricChannel(DiscreteMemorylessChannel):
    """
    Binary symmetric channel (BSC).

    The *binary symmetric channel* (*BSC*) :cite:`Cover.Thomas.06` (Sec. 7.1.4) is a :class:`discrete memoryless channel <DiscreteMemorylessChannel>` with input and output alphabets given by :math:`\\mathcal{X} = \\mathcal{Y} = \\{ 0, 1 \\}`, and transition matrix given by

    .. math::

        p_{Y \mid X} = \\begin{bmatrix} 1-p & p \\\\ p & 1-p \\end{bmatrix},

    where the parameter :math:`p` is called the *crossover probability* of the channel.

    Equivalently, a BSC with crossover probability :math:`p` may be defined by

    .. math::
        Y_n = X_n + Z_n,

    where :math:`Z_n \sim \mathrm{Bernoulli}(p)` are :term:`i.i.d.` random variables.

    Parameters
    ==========
    crossover_probability : `float`, optional
        The channel crossover probability :math:`p`. Must satisfy :math:`0 \leq p \leq 1`. Default value is :code:`0.0`.

    Examples
    ========
    >>> bsc = komm.BinarySymmetricChannel(0.1)
    >>> x = np.random.randint(2, size=10); x
    array([0, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    >>> y = bsc(x); y
    array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0])
    """
    def __init__(self, crossover_probability=0.0):
        super().__init__([[1 - crossover_probability, crossover_probability], [crossover_probability, 1 - crossover_probability]])
        self._crossover_probability = crossover_probability

    @property
    def crossover_probability(self):
        """
        The channel crossover probability :math:`p`.
        """
        return self.crossover_probability

    def capacity(self):
        """
        Returns the channel capacity, given by :math:`C = 1 - \\mathcal{H}(p)`
        :cite:`Cover.Thomas.06` (Sec. 7.1.4).

        Examples
        ========
        >>> bsc = komm.BinarySymmetricChannel(0.25)
        >>> bsc.capacity()
        0.18872187554086717
        """
        return 1.0 - entropy(np.array([self._crossover_probability, 1.0 - self._crossover_probability]))

    def __call__(self, input_sequence):
        error_pattern = bernoulli_iid(self._crossover_probability, input_sequence.size)
        return (input_sequence + error_pattern) % 2

    def __repr__(self):
        args = 'crossover_probability={}'.format(self._crossover_probability)
        return '{}({})'.format(self.__class__.__name__, args)


class BinaryErasureChannel(DiscreteMemorylessChannel):
    """
    Binary erasure channel (BSC).

    The *binary erasure channel* (*BEC*) :cite:`Cover.Thomas.06` (Sec. 7.1.5) is a :class:`discrete memoryless channel <DiscreteMemorylessChannel>` with input alphabet :math:`\\mathcal{X} = \\{ 0, 1 \\}`, output alphabet :math:`\\mathcal{Y} = \\{ 0, 1, 2 \\}`, and transition matrix given by

    .. math::

        p_{Y \mid X} =
        \\begin{bmatrix}
            1 - \\epsilon & 0 & \\epsilon \\\\
            0 & 1 - \\epsilon & \\epsilon
        \\end{bmatrix},

    where the parameter :math:`\\epsilon` is called the *erasure probability* of the channel.

    Parameters
    ==========
    erasure_probability : `float`, optional
        The channel erasure probability :math:`\\epsilon`. Must satisfy :math:`0 \leq \\epsilon \leq 1`. Default value is :code:`0.0`.

    Examples
    ========
    >>> bec = komm.BinaryErasureChannel(0.1)
    >>> x = np.random.randint(2, size=10); x
    array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    >>> y = bec(x); y
    array([1, 1, 1, 2, 0, 0, 1, 0, 1, 0])
    """
    def __init__(self, erasure_probability=0.0):
        super().__init__([[1 - erasure_probability, 0, erasure_probability], [0, 1 - erasure_probability, erasure_probability]])
        self._erasure_probability = erasure_probability

    @property
    def erasure_probability(self):
        """
        The channel erasure probability :math:`\\epsilon`.
        """
        return self._erasure_probability

    def capacity(self):
        """
        Returns the channel capacity, given by :math:`C = 1 - \\epsilon`.

        Examples
        ========
        >>> bec = komm.BinaryErasureChannel(0.25)
        >>> bec.capacity()
        0.75
        """
        return 1.0 - self._erasure_probability

    def __call__(self, input_sequence):
        erasure_pattern = bernoulli_iid(self._erasure_probability, input_sequence.size) == 1
        output_sequence = np.copy(input_sequence)
        output_sequence[erasure_pattern] = 2
        return output_sequence

    def __repr__(self):
        args = 'erasure_probability={}'.format(self._erasure_probability)
        return '{}({})'.format(self.__class__.__name__, args)







def bernoulli_iid(success_probability, size):
    """Bernoulli iid process."""
    return (np.random.rand(size) < success_probability).astype(np.int)


def discrete_memoryless_channel(input_sequence, output_cardinality, transition_matrix):
    output_sequence = [np.random.choice(output_cardinality, p=transition_matrix[input_symbol])
                       for input_symbol in input_sequence]
    return np.array(output_sequence)


def arimoto_blahut(transition_matrix, initial_guess, max_iters, error_tolerance):
    """
    Arimoto--Blahut algorithm for channel capacity :cite:`Cover.Thomas.06` (p. 335).
    """
    p = transition_matrix
    r = initial_guess
    last_r = np.full_like(r, fill_value=np.inf)
    iters = 0
    while iters < max_iters and np.max(np.abs(r - last_r)) > error_tolerance:
        last_r = r
        q = r[np.newaxis].T * p
        q /= np.sum(q, axis=0)
        r = np.product(q ** p, axis=1)
        r /= np.sum(r, axis=0)
        iters += 1
    return r
