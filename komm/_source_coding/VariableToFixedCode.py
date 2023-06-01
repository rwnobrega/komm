import itertools
import numpy as np

from .util import _parse_prefix_free


class VariableToFixedCode:
    """
    Binary (prefix-free) variable-to-fixed length code. Let :math:`\\mathcal{X}` be the alphabet of some discrete source. A *binary variable-to-fixed length code* of code block size :math:`n` is defined by a (possibly partial) decoding mapping :math:`\\mathrm{Dec} : \\{ 0, 1 \\}^n \\to \\mathcal{X}^+`, where :math:`\\mathcal{X}^+` denotes the set of all finite-length, non-empty strings from the source alphabet. Here, for simplicity, the source alphabet is always taken as :math:`\\mathcal{X} = \\{0, 1, \\ldots, |\\mathcal{X} - 1| \\}`. The elements in the image of :math:`\\mathrm{Dec}` are called *sourcewords*.

    Also, we only consider *prefix-free* codes, in which no sourceword is a prefix of any other sourceword.
    """
    def __init__(self, sourcewords):
        """
        Constructor for the class. It expects the following parameters:

        :code:`sourcewords` : :obj:`list` of :obj:`tuple` of :obj:`int`
            The sourcewords of the code. Must be a list of length at most :math:`2^n` containing tuples of integers in :math:`\\mathcal{X}`. The tuple in position :math:`i` of :code:`sourcewords` should be equal to :math:`\\mathrm{Dec}(v)`, where :math:`v` is the :math:`i`-th element in the lexicographic ordering of :math:`\\{ 0, 1 \\}^n`.

        *Note:* The code block size :math:`n` is inferred from :code:`len(sourcewords)`.

        .. rubric:: Examples

        >>> code = komm.VariableToFixedCode(sourcewords=[(1,), (2,), (0,1), (0,2), (0,0,0), (0,0,1), (0,0,2)])
        >>> pprint(code.enc_mapping)
        {(0, 0, 0): (1, 0, 0),
         (0, 0, 1): (1, 0, 1),
         (0, 0, 2): (1, 1, 0),
         (0, 1): (0, 1, 0),
         (0, 2): (0, 1, 1),
         (1,): (0, 0, 0),
         (2,): (0, 0, 1)}
        >>> pprint(code.dec_mapping)
        {(0, 0, 0): (1,),
         (0, 0, 1): (2,),
         (0, 1, 0): (0, 1),
         (0, 1, 1): (0, 2),
         (1, 0, 0): (0, 0, 0),
         (1, 0, 1): (0, 0, 1),
         (1, 1, 0): (0, 0, 2)}
        """
        # TODO: Assert prefix-free
        self._sourcewords = sourcewords
        self._source_cardinality = max(itertools.chain(*sourcewords)) + 1
        self._code_block_size = (len(sourcewords) - 1).bit_length()
        self._enc_mapping = {}
        self._dec_mapping = {}
        for symbols, bits in zip(itertools.product(range(2), repeat=self._code_block_size), sourcewords):
            self._enc_mapping[bits] = tuple(symbols)
            self._dec_mapping[tuple(symbols)] = bits

    @property
    def source_cardinality(self):
        """
        The cardinality :math:`|\\mathcal{X}|` of the source alphabet.
        """
        return self._source_cardinality

    @property
    def code_block_size(self):
        """
        The code block size :math:`n`.
        """
        return self._code_block_size

    @property
    def enc_mapping(self):
        """
        The encoding mapping :math:`\\mathrm{Enc}` of the code.
        """
        return self._enc_mapping

    @property
    def dec_mapping(self):
        """
        The decoding mapping :math:`\\mathrm{Dec}` of the code.
        """
        return self._dec_mapping

    def rate(self, pmf):
        """
        Computes the expected rate :math:`R` of the code, assuming a given :term:`pmf`. It is given in bits per source symbol.

        .. rubric:: Input

        :code:`pmf` : 1D-array of :obj:`float`
            The (first-order) probability mass function :math:`p_X(x)` to be assumed.

        .. rubric:: Output

        :code:`rate` : :obj:`float`
            The expected rate :math:`R` of the code.

        .. rubric:: Examples

        >>> code = komm.VariableToFixedCode([(0,0,0), (0,0,1), (0,1), (1,)])
        >>> code.rate([2/3, 1/3])
        0.9473684210526315
        """
        probabilities = np.array([np.prod([pmf[x] for x in symbols]) for symbols in self._sourcewords])
        lengths = [len(symbols) for symbols in self._sourcewords]
        return self._code_block_size / np.dot(lengths, probabilities)

    def encode(self, symbol_sequence):
        """
        Encodes a given sequence of symbols to its corresponding sequence of bits.

        .. rubric:: Input

        :code:`symbol_sequence` : 1D-array of :obj:`int`
            The sequence of symbols to be encoded. Must be a 1D-array with elements in :math:`\\mathcal{X} = \\{0, 1, \\ldots, |\\mathcal{X} - 1| \\}`.

        .. rubric:: Output

        :code:`bit_sequence` : 1D-array of :obj:`int`
            The sequence of bits corresponding to :code:`symbol_sequence`.

        .. rubric:: Examples

        >>> code = komm.VariableToFixedCode([(0,0,0), (0,0,1), (0,1), (1,)])
        >>> code.encode([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
        """
        return np.array(_parse_prefix_free(symbol_sequence, self._enc_mapping))

    def decode(self, bit_sequence):
        """
        Decodes a given sequence of bits to its corresponding sequence of symbols.

        .. rubric:: Input

        :code:`bit_sequence` : 1D-array of :obj:`int`
            The sequence of bits to be decoded. Must be a 1D-array with elements in :math:`\\{ 0, 1 \\}`.  Its length must be a multiple of :math:`n`.

        .. rubric:: Output

        :code:`symbol_sequence` : 1D-array of :obj:`int`
            The sequence of symbols corresponding to :code:`bits`.

        .. rubric:: Examples

        >>> code = komm.VariableToFixedCode([(0,0,0), (0,0,1), (0,1), (1,)])
        >>> code.decode([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
        array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        """
        bits_reshaped = np.reshape(bit_sequence, newshape=(-1, self._code_block_size))
        return np.concatenate([self._dec_mapping[tuple(bits)] for bits in bits_reshaped])

    def __repr__(self):
        args = 'sourcewords={}'.format(self._sourcewords)
        return '{}({})'.format(self.__class__.__name__, args)
