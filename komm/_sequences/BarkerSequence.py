from .BinarySequence import BinarySequence


class BarkerSequence(BinarySequence):
    r"""
    Barker sequence. A Barker sequence is a binary sequence (:obj:`BinarySequence`) with autocorrelation $R[\ell]$ satisfying $|R[\ell]| \leq 1$, for $\ell \neq 0$. The only known Barker sequences (up to negation and reversion) are shown in the table below.

    | Length $L$ | Barker sequence $b[n]$ |
    | :--------: | ---------------------- |
    | $2$        | $01$ and $00$          |
    | $3$        | $001$                  |
    | $4$        | $0010$ and $0001$      |
    | $5$        | $00010$                |
    | $7$        | $0001101$              |
    | $11$       | $00011101101$          |
    | $13$       | $0000011001010$        |

    References:

        1. https://en.wikipedia.org/wiki/Barker_code
    """

    def __init__(self, length):
        r"""
        Constructor for the class.

        Parameters:

            length (int): Length of the Barker sequence. Must be in the set $\\{ 2, 3, 4, 5, 7, 11, 13 \\}$.

        Examples:

            >>> barker = komm.BarkerSequence(length=13)
            >>> barker.polar_sequence
            array([ 1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1])
            >>> barker.autocorrelation()
            array([13,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1])
        """
        super().__init__(bit_sequence=self._barker_sequence(length))

    def __repr__(self):
        args = "length={}".format(self.length)
        return "{}({})".format(self.__class__.__name__, args)

    @staticmethod
    def _barker_sequence(length):
        return {
            2: [0, 1],
            3: [0, 0, 1],
            4: [0, 0, 1, 0],
            5: [0, 0, 0, 1, 0],
            7: [0, 0, 0, 1, 1, 0, 1],
            11: [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
            13: [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        }[length]
