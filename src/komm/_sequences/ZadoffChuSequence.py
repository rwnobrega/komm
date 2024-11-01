import numpy as np

from .ComplexSequence import ComplexSequence


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
    """

    def __init__(self, length, root_index=1):
        r"""
        Constructor for the class.

        Parameters:
            length (int): The length $L$ of the Zadoff–Chu sequence. Must be an odd integer.

            root_index (Optional[int]): The root index $q$ of the Zadoff–Chu sequence. Must be in $[1:L)$. The default value is $1$.

        Examples:
            >>> zadoff_chu = ZadoffChuSequence(5, root_index=1)
            >>> np.around(zadoff_chu.sequence, decimals=6)  #doctest: +NORMALIZE_WHITESPACE
            array([ 1.      +0.j      ,  0.309017-0.951057j, -0.809017+0.587785j,  0.309017-0.951057j,  1.      +0.j      ])
            >>> np.around(zadoff_chu.cyclic_autocorrelation(normalized=True), decimals=6)
            array([ 1.+0.j, -0.-0.j, -0.-0.j,  0.+0.j, -0.+0.j])
        """
        if length % 2 == 0:
            raise ValueError("The length must be an odd integer.")
        if not 1 <= root_index < length:
            raise ValueError("The root index must satisfy 1 <= root_index < length.")
        self._length = length
        self._root = root_index
        super().__init__(self._zadoff_chu_sequence(length, root_index))

    @staticmethod
    def _zadoff_chu_sequence(length, root_index):
        n = np.arange(length)
        return np.exp(-1j * np.pi * root_index * n * (n + 1) / length)

    @property
    def length(self):
        r"""
        The length $L$ of the Zadoff–Chu sequence.
        """
        return self._length

    @property
    def root_index(self):
        r"""
        The root index $q$ of the Zadoff–Chu sequence.
        """
        return self._root
