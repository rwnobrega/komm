import numpy as np

__all__ = ['PrefixCode', 'HuffmanCode', 'TunstallCode']


class PrefixCode:
    """
    Binary prefix code. Let :math:`\\mathcal{X}` be a finite set, called the *input alphabet*, and let :math:`\\{ 0, 1 \\}^+` denote the set of all the non-empty binary strings. A *binary prefix code* is defined by a mapping :math:`\\mathrm{Enc} : \\mathcal{X} \\to \\{ 0, 1 \\}^+` in which the *prefix-free* (or *instantaneous*) property is satisfied: no codeword in the codebook (the image of the encoding map) is a prefix of any other codeword.  Here, for simplicity, the input alphabet is always taken as :math:`\\mathcal{X} = \\{0, 1, \\ldots, |\\mathcal{X} - 1| \\}`.

    .. rubric:: Examples

    >>> code = komm.PrefixCode([(0,0), (1,0), (1,1), (0,1,0), (0,1,1,0), (0,1,1,1)])
    >>> code.encode([0, 5, 3, 0, 1, 2])
    array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1])
    >>> code.decode([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1])
    array([0, 5, 3, 0, 1, 2])
    """
    def __init__(self, mapping):
        """
        Constructor for the class. It expects the following parameter:

        :code:`mapping` : :obj:`list` of :obj:`tuple` of :obj:`int`
            The encoding mapping :math:`\\mathrm{Enc}` of the prefix code. Must be a list containing tuples of integers in :math:`\\{ 0, 1 \\}`. The tuple in index :math:`x` of :code:`mapping` should be equal to :math:`\\mathrm{Enc}(x)`, for all :math:`x \\in \\mathcal{X}`.
        """
        self._mapping = [tuple(bits) for bits in mapping]
        self._reverse_mapping = {tuple(bits): symbol for (symbol, bits) in enumerate(mapping)}

    @property
    def mapping(self):
        """
        The encoding mapping :math:`\\mathrm{Enc}` of the prefix code.
        """
        return self._mapping

    def average_length(self, pmf):
        """
        Computes the average length :math:`\\mathrm{E}[\\ell]` of the symbol code assuming a given :term:`pmf`. It is given by

        .. math::

            \\mathrm{E}[\\ell] = \\sum_{x \\in \\mathcal{X}} p_X(x) \\ell(x),

        where :math:`p_X(x)` is the :term:`pmf` to be assumed, and :math:`\\ell(x)` is the number of bits in :math:`\\mathrm{Enc}(x)`.

        **Input:**

        :code:`pmf` : 1D-array of :obj:`float`
            The probability mass function :math:`p_X(x)` to be assumed.

        **Output:**

        :code:`average_length` : :obj:`float`
            The average length :math:`\\mathrm{E}[\\ell]` of the symbol code assuming the given :term:`pmf`.
        """
        return np.dot([len(bit_sequence) for bit_sequence in self._mapping], pmf) / np.sum(pmf)

    def encode(self, symbols):
        """
        Encodes a given sequence of symbols to its corresponding sequence of bits.

        **Input:**

        :code:`symbols` : 1D-array of :obj:`int`
            The sequence of symbols to be encoded. Must be a 1D-array with elements in :math:`\\mathcal{X} = \\{0, 1, \\ldots, |\\mathcal{X} - 1| \\}`

        **Output:**

        :code:`bits` : 1D-array of :obj:`int`
            The sequence of bits corresponding to :code:`symbols`.
        """
        return np.concatenate([self._mapping[symbol] for symbol in symbols])

    def decode(self, bits):
        """
        Decodes a given sequence of bits to its corresponding sequence of symbols.

        **Input:**

        :code:`bits` : 1D-array of :obj:`int`
            The sequence of bits to be decoded. Must be a 1D-array with elements in :math:`\\{ 0, 1 \\}`.

        **Output:**

        :code:`symbols` : 1D-array of :obj:`int`
            The sequence of symbols corresponding to :code:`bits`.
        """
        symbols = []
        i = 0
        while i < len(bits):
            j = 1
            while True:
                bit_sequence = tuple(bits[i : i + j])
                if bit_sequence in self._reverse_mapping:
                    break
                else:
                    j += 1
            symbols.append(self._reverse_mapping[bit_sequence])
            i += j
        return np.array(symbols)

    def __repr__(self):
        args = 'mapping={}'.format(self._mapping)
        return '{}({})'.format(self.__class__.__name__, args)


class HuffmanCode:
    """
    Huffman code [Not implemented yet].
    """
    pass


class TunstallCode:
    """
    Tunstall code [Not implemented yet].
    """
    pass
