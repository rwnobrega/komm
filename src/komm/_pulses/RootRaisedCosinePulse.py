import numpy as np

from .FormattingPulse import FormattingPulse


class RootRaisedCosinePulse(FormattingPulse):
    r"""
    Root raised cosine pulse. It is a formatting pulse with impulse response given by
    $$
        h(t) = \frac{\sin[\pi (1 - \alpha) t] + 4 \alpha t \cos[\pi (1 + \alpha) t]}{\pi t [1 - (4 \alpha t)^2]},
    $$
    where $\alpha$ is the *roll-off factor*. The root raised cosine pulse is depicted below for $\alpha = 0.25$, and for $\alpha = 0.75$.

    <div class="centered" markdown>
      <span>
        ![Root raised cosine pulse with roll-off factor 0.25.](/figures/pulse_root_raised_cosine_25.svg)
      </span>
      <span>
        ![Root raised cosine pulse with roll-off factor 0.75.](/figures/pulse_root_raised_cosine_75.svg)
      </span>
    </div>
    """

    def __init__(self, rolloff, length_in_symbols):
        r"""
        Constructor for the class.

        Parameters:
            rolloff (float): The roll-off factor $\alpha$ of the pulse. Must satisfy $0 \leq \alpha \leq 1$.

            length_in_symbols (int): The length (span) of the truncated impulse response, in symbols.

        Examples:
            >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.25, length_in_symbols=16)

            >>> pulse = komm.RootRaisedCosinePulse(rolloff=0.75, length_in_symbols=16)
        """
        a = self._rolloff = float(rolloff)
        L = self._length_in_symbols = int(length_in_symbols)

        def impulse_response(t):
            t += 1e-8
            return (
                np.sin(np.pi * (1 - a) * t) + 4 * a * t * np.cos(np.pi * (1 + a) * t)
            ) / (np.pi * t * (1 - (4 * a * t) ** 2))

        def frequency_response(f):
            f1 = (1 - a) / 2
            f2 = (1 + a) / 2
            H = 1.0 * (abs(f) < f1)
            if a > 0:
                H += np.sqrt(
                    (f1 < abs(f) < f2)
                    * (0.5 + 0.5 * np.cos((np.pi * (abs(f) - f1)) / (f2 - f1)))
                )
            return H

        super().__init__(impulse_response, frequency_response, interval=(-L / 2, L / 2))

    @property
    def rolloff(self):
        r"""
        The roll-off factor $\alpha$ of the pulse.
        """
        return self._rolloff

    @property
    def length_in_symbols(self):
        r"""
        The length (span) of the truncated impulse response.
        """
        return self._length_in_symbols

    def __repr__(self):
        args = "rolloff={}, length_in_symbols={}".format(
            self._rolloff, self._length_in_symbols
        )
        return "{}({})".format(self.__class__.__name__, args)
