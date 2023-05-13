import numpy as np

from ._util import \
    _entropy


__all__ = ['DiscreteMemorylessSource']


class DiscreteMemorylessSource:
    """
    Discrete memoryless source (DMS). It is defined by an *alphabet* :math:`\\mathcal{X}` and a *probability mass function* (:term:`pmf`) :math:`p_X`. Here, for simplicity, the alphabet is always taken as :math:`\\mathcal{X} = \\{ 0, 1, \\ldots, |\\mathcal{X}| - 1 \\}`. The :term:`pmf` :math:`p_X` gives the probability of the source emitting the symbol :math:`X = x`.

    To invoke the source, call the object giving the number of symbols to be emitted as parameter (see example below).
    """
    def __init__(self, pmf):
        """
        Constructor for the class. It expects the following parameter:

        :code:`pmf` : 1D-array of :obj:`float`
            The source probability mass function :math:`p_X`. The element in position :math:`x \\in \\mathcal{X}` must be equal to :math:`p_X(x)`.

        .. rubric:: Examples

        >>> dms = komm.DiscreteMemorylessSource([0.5, 0.4, 0.1])
        >>> dms(10)  #doctest:+SKIP
        array([1, 2, 1, 0, 0, 1, 1, 0, 1, 1])
        """
        self.pmf = pmf

    @property
    def pmf(self):
        """
        The source probability mass function :math:`p_X`. This is a read-and-write property.
        """
        return self._pmf

    @pmf.setter
    def pmf(self, value):
        self._pmf = np.array(value, dtype=float)
        self._cardinality = self._pmf.size

    @property
    def cardinality(self):
        """
        The cardinality :math:`|\\mathcal{X}|` of the source alphabet. This property is read-only.
        """
        return self._cardinality

    def entropy(self):
        """
        Returns the source entropy :math:`\\mathrm{H}(X)`.

        .. rubric:: Examples

        >>> dms = komm.DiscreteMemorylessSource([1/2, 1/4, 1/8, 1/8])
        >>> dms.entropy()
        1.75
        """
        return _entropy(self._pmf)

    def __call__(self, size):
        return np.random.choice(self._cardinality, p=self._pmf, size=size)

    def __repr__(self):
        args = 'pmf={}'.format(self._pmf.tolist())
        return '{}({})'.format(self.__class__.__name__, args)
