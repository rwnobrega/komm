import numpy as np

from .._util import _entropy
from .._validation import validate, validate_pmf


class DiscreteMemorylessSource:
    r"""
    Discrete memoryless source (DMS). It is defined by an *alphabet* $\mathcal{X}$ and a *probability mass function* (pmf) $p_X$. Here, for simplicity, the alphabet is always taken as $\mathcal{X} = \\{ 0, 1, \ldots, |\mathcal{X}| - 1 \\}$. The pmf $p_X$ gives the probability of the source emitting the symbol $X = x$.

    To invoke the source, call the object giving the number of symbols to be emitted as parameter (see example in the constructor below).
    """

    @validate(pmf=validate_pmf)
    def __init__(self, pmf):
        r"""
        Constructor for the class.

        Parameters:

            pmf (Array1D[float]): The source probability mass function $p_X$. The element in position $x \in \mathcal{X}$ must be equal to $p_X(x)$.

        Examples:

            >>> np.random.seed(42)
            >>> dms = komm.DiscreteMemorylessSource([0.5, 0.4, 0.1])
            >>> dms(10)
            array([0, 2, 1, 1, 0, 0, 0, 1, 1, 1])
        """
        self._pmf = pmf

    @property
    def pmf(self):
        r"""
        The source probability mass function $p_X$.
        """
        return self._pmf

    @pmf.setter
    def pmf(self, value):
        self._pmf = np.array(value, dtype=float)

    @property
    def cardinality(self):
        r"""
        The cardinality $|\mathcal{X}|$ of the source alphabet.
        """
        return self._pmf.size

    def entropy(self, base=2.0):
        r"""
        Returns the source entropy $\mathrm{H}(X)$. See [`komm.entropy`](/ref/entropy) for more details.

        Parameters:

            base (Optional[float | str]): See [`komm.entropy`](/ref/entropy). The default value is $2.0$.

        Examples:

            >>> dms = komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
            >>> dms.entropy()
            1.75
            >>> dms.entropy(base=4)
            0.875
        """
        return _entropy(self._pmf, base=base)

    def __call__(self, size):
        return np.random.choice(self._pmf.size, p=self._pmf, size=size)

    def __repr__(self):
        args = "pmf={}".format(self._pmf.tolist())
        return "{}({})".format(self.__class__.__name__, args)
