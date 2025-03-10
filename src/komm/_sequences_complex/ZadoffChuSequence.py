from .ComplexSequence import ComplexSequence
from .sequences import zadoff_chu_sequence


class ZadoffChuSequence(ComplexSequence):
    r"""
    Zadoff–Chu sequence. It is a periodic, [complex sequence](/ref/ComplexSequence) given by
    $$
        z_{L,q}[n] = \mathrm{e}^{-\mathrm{j} \pi q n (n + 1) / L},
    $$
    where $L$ is the *length* (and period) of the sequence (which must be an odd integer) and $q \in [1:L)$ is called the *root index* of the sequence.

    Zadoff–Chu sequences have the following properties:

    1. *Constant amplitude:* The magnitude of the sequence satisfies
    $$
        |z_{L,q}[n]| = 1, \quad \forall n.
    $$

    2. *Zero autocorrelation:* If $q$ is coprime to $L$, then the cyclic autocorrelation of $z_{L,q}$ satisfies
    $$
        \tilde{R}\_{z_{L,q}}[\ell] = 0, \quad \forall \ell \neq 0 \mod L.
    $$

    3. *Constant cross-correlation:* If $|q' - q|$ is coprime to $L$, then the magnitude of the cyclic cross-correlation of $z_{L,q}$ and $z_{L,q'}$ satisfies
    $$
        |\tilde{R}\_{z_{L,q}, z_{L,q'}}[\ell]| = \sqrt{L}, \quad \forall \ell.
    $$

    For more details, see <cite>And22</cite>.

    Notes:
        - Theses sequences are also called *Frank–Zadoff–Chu* sequences.

    Parameters:
        length: The length $L$ of the Zadoff–Chu sequence. Must be an odd integer.
        root_index: The root index $q$ of the Zadoff–Chu sequence. Must be in $[1:L)$. The default value is $1$.

    Examples:
        >>> zadoff_chu = ZadoffChuSequence(5, root_index=1)
        >>> zadoff_chu.sequence.round(6)
        array([ 1.      +0.j      ,  0.309017-0.951057j, -0.809017+0.587785j,  0.309017-0.951057j,  1.      +0.j      ])
        >>> zadoff_chu.cyclic_autocorrelation(normalized=True).round(6)
        array([ 1.+0.j, -0.-0.j, -0.-0.j,  0.+0.j, -0.+0.j])
    """

    def __init__(self, length: int, root_index: int = 1) -> None:
        if length % 2 == 0:
            raise ValueError("'length' must be an odd integer")
        if not 1 <= root_index < length:
            raise ValueError("'root_index' must be in [1 : length)")
        super().__init__(sequence=zadoff_chu_sequence(length, root_index))
