from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import base


@dataclass(eq=False)
class TransmitFilter:
    r"""
    Transmit filter (pulse shaping). Given a sequence of $N$ real or complex symbols $x[n]$, this filter outputs samples of the signal
    $$
        x(t) = \sum_{n=0}^{N-1} x[n] h(t - n),
    $$
    where $h(t)$ is the waveform of a given [pulse](/ref/Pulse), and the samples of the output signal are taken at an integer rate of $\beta$ samples per symbol. Note that the symbol interval is normalized to $1$.

    The time span of $x(t)$ is given by $[ n_0, n_1 + N - 1 )$, where $[ n_0, n_1 )$ is the integer-bounded time span of $h(t)$. In turn, $n_0$ and $n_1$ depend on the support of $h(t)$:

    - If $h(t)$ has finite support $[ t_0, t_1 ]$, then $n_0 = \lfloor t_0 \rfloor$ and $n_1 = \lceil t_1 \rceil$.

    - If $h(t)$ has infinite support, then $n_0 = -L/2$ and $n_1 = L/2$, where $L$ is a given even positive integer, called the _truncation window length_.

    Attributes:
        pulse: The pulse whose waveform is $h(t)$.
        samples_per_symbol: The number $\beta$ of samples (of the output) per symbol (of the input). Must be a positive integer.
        truncation: The truncation window length $L$. Only applies to infinite-duration pulses. Must be an even positive integer. The default value is `32`.
    """

    pulse: base.Pulse
    samples_per_symbol: int
    truncation: int | None = None

    def _pulse_support_kind(self) -> Literal["finite", "infinite", "semi-infinite"]:
        support = self.pulse.support
        if support == (-np.inf, np.inf):
            return "infinite"
        elif support[0] != -np.inf and support[1] != np.inf:
            return "finite"
        else:
            return "semi-infinite"

    def __post_init__(self) -> None:
        if self._pulse_support_kind() == "finite":
            if self.truncation is not None:
                raise ValueError("'truncation' only applies to infinite-support pulses")
        elif self._pulse_support_kind() == "infinite":
            if self.truncation is None:
                self.truncation = 32
            elif self.truncation <= 0 or self.truncation % 2 != 0:
                raise ValueError("'truncation' must be an even positive integer")
        else:  # self._pulse_support_kind() == "semi-infinite"
            raise ValueError("pulses with semi-infinite support are not supported")

    @cached_property
    def pulse_time_span(self) -> tuple[int, int]:
        r"""
        The integer-bounded time span $[ n_0, n_1 )$ of the pulse waveform $h(t)$.

        Examples:
            >>> pulse = komm.RectangularPulse(0.25)
            >>> pulse.support
            (0.0, 0.25)
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=3)
            >>> tx_filter.pulse_time_span
            (0, 1)

            >>> pulse = komm.SincPulse()
            >>> pulse.support
            (-inf, inf)
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=4, truncation=4)
            >>> tx_filter.pulse_time_span
            (-2, 2)
        """
        if self._pulse_support_kind() == "infinite":
            assert self.truncation is not None  # Guaranteed
            t_min_h = -self.truncation // 2
            t_max_h = self.truncation // 2
        else:  # self._pulse_support_kind() == "finite"
            support = self.pulse.support
            t_min_h = int(np.floor(support[0]))
            t_max_h = int(np.ceil(support[1]))
        return t_min_h, t_max_h

    def _time(self, num_symbols: int) -> npt.NDArray[np.floating]:
        # The time axis of the output signal considering 'num_symbols' input symbols.
        t_min_h, t_max_h = self.pulse_time_span
        t_min_x, t_max_x = t_min_h, t_max_h + num_symbols - 1
        beta = self.samples_per_symbol
        t = np.arange(t_min_x * beta, t_max_x * beta) / beta
        return t

    @cached_property
    def taps(self) -> npt.NDArray[np.floating]:
        r"""
        The FIR filter taps of the transmit filter.

        Examples:
            >>> pulse = komm.RectangularPulse(width=0.25)
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=3)
            >>> tx_filter.taps
            array([1., 0., 0.])

            >>> pulse = komm.SincPulse()
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=4, truncation=4)
            >>> tx_filter.taps.reshape((-1, 4)).round(6)
            array([[-0.      , -0.128617, -0.212207, -0.180063],
                   [ 0.      ,  0.300105,  0.63662 ,  0.900316],
                   [ 1.      ,  0.900316,  0.63662 ,  0.300105],
                   [ 0.      , -0.180063, -0.212207, -0.128617]])
        """
        return self.pulse.waveform(self._time(1))

    def time(self, input: npt.ArrayLike) -> npt.NDArray[np.floating]:
        r"""
        Convenience function to generate the time axis of the output signal given the input symbols.

        Parameters:
            input: The input symbols $x[n]$, of length $N$.

        Returns:
            t: The time axis of the output signal, of length $(N + n_1 - n_0 - 1) \beta$.

        Examples:
            >>> pulse = komm.RectangularPulse()
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=3)
            >>> tx_filter.time([1.0, -1.0, 1.0, 1.0]).round(2)
            array([0.  , 0.33, 0.67, 1.  , 1.33, 1.67, 2.  , 2.33, 2.67, 3.  , 3.33, 3.67])

            >>> pulse = komm.SincPulse()
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=4, truncation=4)
            >>> tx_filter.time([1.0, -1.0, 1.0, 1.0]).reshape((-1, 4))
            array([[-2.  , -1.75, -1.5 , -1.25],
                   [-1.  , -0.75, -0.5 , -0.25],
                   [ 0.  ,  0.25,  0.5 ,  0.75],
                   [ 1.  ,  1.25,  1.5 ,  1.75],
                   [ 2.  ,  2.25,  2.5 ,  2.75],
                   [ 3.  ,  3.25,  3.5 ,  3.75],
                   [ 4.  ,  4.25,  4.5 ,  4.75]])
        """
        input = np.asarray(input)
        return self._time(input.size)

    def __call__(
        self, input: npt.ArrayLike
    ) -> npt.NDArray[np.floating | np.complexfloating]:
        r"""
        Process the input symbols through the transmit filter.

        Parameters:
            input: The input symbols $x[n]$, of length $N$.

        Returns:
            output: The samples of the output signal $x(t)$, of length $(N + n_1 - n_0 - 1) \beta$.

        Examples:
            >>> pulse = komm.RectangularPulse(width=1.0)
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=3)
            >>> tx_filter([1.0, -1.0, 1.0, 1.0])
            array([ 1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1.])

            >>> pulse = komm.RectangularPulse(width=0.25)
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=3)
            >>> tx_filter([1.0, -1.0, 1.0, 1.0])
            array([ 1.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.])

            >>> pulse = komm.SincPulse()
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=4, truncation=4)
            >>> tx_filter([1.0, -1.0, 1.0, 1.0]).reshape((-1, 4)).round(6)
            array([[-0.      , -0.128617, -0.212207, -0.180063],
                   [ 0.      ,  0.428722,  0.848826,  1.08038 ],
                   [ 1.      ,  0.471594, -0.212207, -0.780274],
                   [-1.      , -0.908891, -0.424413,  0.291531],
                   [ 1.      ,  1.380485,  1.485446,  1.329038],
                   [ 1.      ,  0.720253,  0.424413,  0.171489],
                   [ 0.      , -0.180063, -0.212207, -0.128617]])

            >>> pulse = komm.RectangularPulse()
            >>> tx_filter = komm.TransmitFilter(pulse=pulse, samples_per_symbol=4, truncation=4)
            Traceback (most recent call last):
            ...
            ValueError: 'truncation' only applies to infinite-support pulses
        """
        input = np.asarray(input)
        beta = self.samples_per_symbol
        input_interp = np.zeros((input.size - 1) * beta + 1, dtype=input.dtype)
        input_interp[::beta] = input
        output = np.convolve(self.taps, input_interp)
        return output
