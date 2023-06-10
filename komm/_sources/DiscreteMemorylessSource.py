import numpy as np

from .._util import _entropy


class DiscreteMemorylessSource:
    r"""
    Discrete memoryless source (DMS). It is defined by an *alphabet* $\mathcal{X}$ and a *probability mass function* (:term:`pmf`) $p_X$. Here, for simplicity, the alphabet is always taken as $\mathcal{X} = \\{ 0, 1, \ldots, |\mathcal{X}| - 1 \\}$. The :term:`pmf` $p_X$ gives the probability of the source emitting the symbol $X = x$.

    To invoke the source, call the object giving the number of symbols to be emitted as parameter (see example in the constructor below).
    """

    def __init__(self, pmf):
        r"""
        Constructor for the class.

        Parameters:

            pmf (1D-array of :obj:`float`): The source probability mass function $p_X$. The element in position $x \in \mathcal{X}$ must be equal to $p_X(x)$.

        Examples:

            >>> dms = komm.DiscreteMemorylessSource([0.5, 0.4, 0.1])
            >>> dms(10)  #doctest: +SKIP
            array([1, 2, 1, 0, 0, 1, 1, 0, 1, 1])
        """
        self.pmf = pmf

    @property
    def pmf(self):
        r"""
        The source probability mass function $p_X$. This is a read-and-write property.
        """
        return self._pmf

    @pmf.setter
    def pmf(self, value):
        self._pmf = np.array(value, dtype=float)
        self._cardinality = self._pmf.size

    @property
    def cardinality(self):
        r"""
        The cardinality $|\mathcal{X}|$ of the source alphabet. This property is read-only.
        """
        return self._cardinality

    def entropy(self, base=2.0):
        r"""
        Returns the source entropy $\mathrm{H}(X)$.

        Parameters:

            base (:obj:`float` or :obj:`str`, optional): The base of the logarithm to be used. It must be a positive float or the string :code:`'e'`. The default value is :code:`2.0`.

        Examples:

            >>> dms = komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
            >>> dms.entropy()
            1.75
            >>> dms.entropy(base=4)
            0.875
        """
        return _entropy(self._pmf, base=base)

    def __call__(self, size):
        return np.random.choice(self._cardinality, p=self._pmf, size=size)

    def __repr__(self):
        args = "pmf={}".format(self._pmf.tolist())
        return "{}({})".format(self.__class__.__name__, args)
