from .BinarySequence import BinarySequence


class BarkerSequence(BinarySequence):
    """
    Barker sequence. A Barker sequence is a binary sequence (:obj:`BinarySequence`) with autocorrelation :math:`R[\\ell]` satisfying :math:`|R[\\ell]| \\leq 1`, for :math:`\\ell \\neq 0`. The only known Barker sequences (up to negation and reversion) are shown in the table below.

    ================  =============================
    Length :math:`L`  Barker sequence :math:`b[n]`
    ================  =============================
    :math:`2`         :math:`01` and :math:`00`
    :math:`3`         :math:`001`
    :math:`4`         :math:`0010` and :math:`0001`
    :math:`5`         :math:`00010`
    :math:`7`         :math:`0001101`
    :math:`11`        :math:`00011101101`
    :math:`13`        :math:`0000011001010`
    ================  =============================

    [1] https://en.wikipedia.org/wiki/Barker_code
    """

    def __init__(self, length):
        """
        Constructor for the class. It expects the following parameter:

        :code:`length` : :obj:`int`
            Length of the Barker sequence. Must be in the set :math:`\\{ 2, 3, 4, 5, 7, 11, 13 \\}`.

        .. rubric:: Examples

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
