from .BinarySequence import BinarySequence
from .sequences import barker_sequence


class BarkerSequence(BinarySequence):
    r"""
    Barker sequence. A Barker sequence is a [binary sequence](/ref/BinarySequence) with autocorrelation $R[\ell]$ satisfying $|R[\ell]| \leq 1$, for $\ell \neq 0$. The only known Barker sequences (up to negation and reversion) are shown in the table below. For more details, see [Wikipedia: Barker code](https://en.wikipedia.org/wiki/Barker_code).

    | Length $L$ | Barker sequence $b[n]$ |
    | :--------: | ---------------------- |
    | $2$        | $01$ and $00$          |
    | $3$        | $001$                  |
    | $4$        | $0010$ and $0001$      |
    | $5$        | $00010$                |
    | $7$        | $0001101$              |
    | $11$       | $00011101101$          |
    | $13$       | $0000011001010$        |

    Parameters:
        length: Length of the Barker sequence. Must be in the set $\\{ 2, 3, 4, 5, 7, 11, 13 \\}$.

    Examples:
        >>> barker = komm.BarkerSequence(length=13)
        >>> barker.polar_sequence
        array([ 1,  1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1])
        >>> barker.autocorrelation()
        array([13,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1])
    """

    def __init__(self, length: int) -> None:
        allowed_lengths = {2, 3, 4, 5, 7, 11, 13}
        if length not in allowed_lengths:
            raise ValueError(f"'length' must be in {allowed_lengths}")
        super().__init__(bit_sequence=barker_sequence(length))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={self.length})"
