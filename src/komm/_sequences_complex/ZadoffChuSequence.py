from .._util.validators import validate_integer
from .ComplexSequence import ComplexSequence
from .sequences import zadoff_chu_sequence


class ZadoffChuSequence(ComplexSequence):
    r"""
    Zadoff–Chu sequence. It is a periodic, [complex sequence](/ref/ComplexSequence) given by
    $$
        z_{L,q}[n] = \mathrm{e}^{-\mathrm{j} \pi q n (n + 1) / L},
    $$
    where $L$ is the *length* (and period) of the sequence, an odd integer with $L \geq 3$, and $q \in [1:L)$ is called the *root index* of the sequence.

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
        - These sequences are also called *Frank–Zadoff–Chu* sequences.

    Parameters:
        length: The length $L$ of the Zadoff–Chu sequence. Must be an odd integer satisfying $L \geq 3$.
        root_index: The root index $q$ of the Zadoff–Chu sequence. Must be in $[1:L)$. The default value is $1$.

    Examples:
        >>> zadoff_chu = komm.ZadoffChuSequence(5, root_index=1)
        >>> zadoff_chu.sequence.round(6)
        array([ 1.      +0.j      ,  0.309017-0.951057j, -0.809017+0.587785j,
                0.309017-0.951057j,  1.      +0.j      ])
        >>> zadoff_chu.cyclic_autocorrelation(normalized=True).round(15) + 0.0
        array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    """

    def __init__(self, length: int, root_index: int = 1) -> None:
        length = validate_integer(length, low=3)
        if not length % 2 == 1:
            raise ValueError(f"'length' must be odd (got {length})")
        root_index = validate_integer(
            root_index, low=1, high=length, rule="1 <= root_index < length"
        )
        self.root_index = root_index
        super().__init__(sequence=zadoff_chu_sequence(length, root_index))

    def __repr__(self) -> str:
        args = f"length={self.length}, root_index={self.root_index}"
        return f"{self.__class__.__name__}({args})"
