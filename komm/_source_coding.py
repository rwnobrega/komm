import heapq
import itertools

import numpy as np

__all__ = ['FixedToVariableCode', 'VariableToFixedCode',
           'HuffmanCode', 'TunstallCode']


class FixedToVariableCode:
    """
    Binary (prefix-free) fixed-to-variable length code. Let :math:`\\mathcal{X}` be the alphabet of some discrete source. A *binary fixed-to-variable length code* of source block size :math:`k` is defined by an encoding mapping :math:`\\mathrm{Enc} : \\mathcal{X}^k \\to \\{ 0, 1 \\}^+`, where :math:`\\{ 0, 1 \\}^+` denotes the set of all finite-length, non-empty binary strings. Here, for simplicity, the source alphabet is always taken as :math:`\\mathcal{X} = \\{0, 1, \\ldots, |\\mathcal{X} - 1| \\}`. The elements in the image of :math:`\\mathrm{Enc}` are called *codewords*.

    Also, we only consider *prefix-free* codes, in which no codeword is a prefix of any other codeword.
    """
    def __init__(self, codewords, source_cardinality=None):
        """
        Constructor for the class. It expects the following parameters:

        :code:`codewords` : :obj:`list` of :obj:`tuple` of :obj:`int`
            The codewords of the code. Must be a list of length :math:`|\\mathcal{X}|^k` containing tuples of integers in :math:`\\{ 0, 1 \\}`. The tuple in position :math:`i` of :code:`codewords` should be equal to :math:`\\mathrm{Enc}(u)`, where :math:`u` is the :math:`i`-th element in the lexicographic ordering of :math:`\\mathcal{X}^k`.

        :code:`source_cardinality` : :obj:`int`, optional
            The cardinality :math:`|\\mathcal{X}|` of the source alphabet. The default value is :code:`len(codewords)`, yielding a source block size :math:`k = 1`.

        *Note:* The source block size :math:`k` is inferred from :code:`len(codewords)` and :code:`source_cardinality`.

        .. rubric:: Examples

        >>> code = komm.FixedToVariableCode(codewords=[(0,), (1,0), (1,1)])
        >>> pprint(code.enc_mapping)
        {(0,): (0,), (1,): (1, 0), (2,): (1, 1)}
        >>> pprint(code.dec_mapping)
        {(0,): (0,), (1, 0): (1,), (1, 1): (2,)}

        >>> code = komm.FixedToVariableCode(codewords=[(0,), (1,0,0), (1,1), (1,0,1)], source_cardinality=2)
        >>> pprint(code.enc_mapping)
        {(0, 0): (0,), (0, 1): (1, 0, 0), (1, 0): (1, 1), (1, 1): (1, 0, 1)}
        >>> pprint(code.dec_mapping)
        {(0,): (0, 0), (1, 0, 0): (0, 1), (1, 0, 1): (1, 1), (1, 1): (1, 0)}
        """
        # TODO: Assert prefix-free
        self._codewords = codewords
        self._source_cardinality = len(codewords) if source_cardinality is None else int(source_cardinality)
        self._source_block_size = 1
        while self._source_cardinality ** self._source_block_size < len(codewords):
            self._source_block_size += 1

        if self._source_cardinality ** self._source_block_size != len(codewords):
            raise ValueError("Invalid number of codewords")

        self._enc_mapping = {}
        self._dec_mapping = {}
        for symbols, bits in zip(itertools.product(range(self._source_cardinality), repeat=self._source_block_size), codewords):
            self._enc_mapping[symbols] = tuple(bits)
            self._dec_mapping[tuple(bits)] = symbols

    @property
    def source_cardinality(self):
        """
        The cardinality :math:`|\\mathcal{X}|` of the source alphabet.
        """
        return self._source_cardinality

    @property
    def source_block_size(self):
        """
        The source block size :math:`k`.
        """
        return self._source_block_size

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

        >>> code = komm.FixedToVariableCode([(0,), (1,0), (1,1)])
        >>> code.rate([0.5, 0.25, 0.25])
        1.5
        """
        probabilities = np.array([np.prod(ps) for ps in itertools.product(pmf, repeat=self._source_block_size)])
        lengths = [len(bits) for bits in self._codewords]
        return np.dot(lengths, probabilities) / self._source_block_size

    def encode(self, symbol_sequence):
        """
        Encodes a given sequence of symbols to its corresponding sequence of bits.

        .. rubric:: Input

        :code:`symbol_sequence` : 1D-array of :obj:`int`
            The sequence of symbols to be encoded. Must be a 1D-array with elements in :math:`\\mathcal{X} = \\{0, 1, \\ldots, |\\mathcal{X} - 1| \\}`. Its length must be a multiple of :math:`k`.

        .. rubric:: Output

        :code:`bit_sequence` : 1D-array of :obj:`int`
            The sequence of bits corresponding to :code:`symbol_sequence`.

        .. rubric:: Examples

        >>> code = komm.FixedToVariableCode([(0,), (1,0), (1,1)])
        >>> code.encode([1, 0, 1, 0, 2, 0])
        array([1, 0, 0, 1, 0, 0, 1, 1, 0])
        """
        symbols_reshaped = np.reshape(symbol_sequence, newshape=(-1, self._source_block_size))
        return np.concatenate([self._enc_mapping[tuple(symbols)] for symbols in symbols_reshaped])

    def decode(self, bit_sequence):
        """
        Decodes a given sequence of bits to its corresponding sequence of symbols.

        .. rubric:: Input

        :code:`bit_sequence` : 1D-array of :obj:`int`
            The sequence of bits to be decoded. Must be a 1D-array with elements in :math:`\\{ 0, 1 \\}`.

        .. rubric:: Output

        :code:`symbol_sequence` : 1D-array of :obj:`int`
            The sequence of symbols corresponding to :code:`bits`.

        .. rubric:: Examples

        >>> code = komm.FixedToVariableCode([(0,), (1,0), (1,1)])
        >>> code.decode([1, 0, 0, 1, 0, 0, 1, 1, 0])
        array([1, 0, 1, 0, 2, 0])
        """
        return np.array(_parse_prefix_free(bit_sequence, self._dec_mapping))

    def __repr__(self):
        args = 'codewords={}'.format(self._codewords)
        return '{}({})'.format(self.__class__.__name__, args)


class HuffmanCode(FixedToVariableCode):
    """
    Huffman code. It is an optimal (minimal expected rate) fixed-to-variable length code (:class:`FixedToVariableCode`) for a given probability mass function.

    .. rubric:: Examples

    >>> code = komm.HuffmanCode([0.7, 0.15, 0.15])
    >>> pprint(code.enc_mapping)
    {(0,): (0,), (1,): (1, 1), (2,): (1, 0)}

    >>> code = komm.HuffmanCode([0.7, 0.15, 0.15], source_block_size=2)
    >>> pprint(code.enc_mapping)
    {(0, 0): (1,),
     (0, 1): (0, 0, 0, 0),
     (0, 2): (0, 1, 1),
     (1, 0): (0, 1, 0),
     (1, 1): (0, 0, 0, 1, 1, 1),
     (1, 2): (0, 0, 0, 1, 1, 0),
     (2, 0): (0, 0, 1),
     (2, 1): (0, 0, 0, 1, 0, 1),
     (2, 2): (0, 0, 0, 1, 0, 0)}
    """
    def __init__(self, pmf, source_block_size=1, policy='high'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`pmf` : 1D-array of :obj:`float`
            The probability mass function used to construct the code.

        :code:`source_block_size` : :obj:`int`, optional
            The source block size :math:`k`. The default value is :math:`k = 1`.

        :code:`policy` : :obj:`str`, optional
            The policy to be used when constructing the code. It must be either :code:`'high'` (move combined symbols as high as possible) or :code:`'low'` (move combined symbols as low as possible). The default value is :code:`'high'`.
        """
        self._pmf = np.array(pmf)
        self._policy = policy

        if policy not in ['high', 'low']:
            raise ValueError("Parameter 'policy' must be in {'high', 'low'}")

        super().__init__(codewords=HuffmanCode._huffman_algorithm(pmf, source_block_size, policy),
                         source_cardinality=self._pmf.size)

    @property
    def pmf(self):
        """
        The probability mass function used to construct the code. This property is read-only.
        """
        return self._pmf

    @staticmethod
    def _huffman_algorithm(pmf, source_block_size, policy):
        class Node:
            def __init__(self, index, probability):
                self.index = index
                self.probability = probability
                self.parent = None
                self.bit = None
            def __lt__(self, other):
                if policy == 'high':
                    return (self.probability, self.index) < (other.probability, other.index)
                elif policy == 'low':
                    return (self.probability, -self.index) < (other.probability, -other.index)

        tree = [Node(i, np.prod(probs)) for (i, probs) in enumerate(itertools.product(pmf, repeat=source_block_size))]
        queue = [node for node in tree]
        heapq.heapify(queue)
        while len(queue) > 1:
            node1 = heapq.heappop(queue)
            node0 = heapq.heappop(queue)
            node1.bit = 1
            node0.bit = 0
            node = Node(index=len(tree), probability=node0.probability + node1.probability)
            node0.parent = node1.parent = node.index
            heapq.heappush(queue, node)
            tree.append(node)

        codewords = []
        for symbol in range(len(pmf)**source_block_size):
            node = tree[symbol]
            bits = []
            while node.parent is not None:
                bits.insert(0, node.bit)
                node = tree[node.parent]
            codewords.append(tuple(bits))

        return codewords

    def __repr__(self):
        args = 'pmf={}, source_block_size={}'.format(self._pmf.tolist(), self._source_block_size)
        return '{}({})'.format(self.__class__.__name__, args)


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


class TunstallCode(VariableToFixedCode):
    """
    Tunstall code. It is an optimal (minimal expected rate) variable-to-fixed length code (:class:`VariableToFixedCode`) for a given probability mass function.

    .. rubric:: Examples

    >>> code = komm.TunstallCode([0.6, 0.3, 0.1], code_block_size=3)
    >>> pprint(code.enc_mapping)
    {(0, 0, 0): (0, 0, 0),
     (0, 0, 1): (0, 0, 1),
     (0, 0, 2): (0, 1, 0),
     (0, 1): (0, 1, 1),
     (0, 2): (1, 0, 0),
     (1,): (1, 0, 1),
     (2,): (1, 1, 0)}
    """
    def __init__(self, pmf, code_block_size):
        """
        Constructor for the class. It expects the following parameters:

        :code:`pmf` : 1D-array of :obj:`float`
            The probability mass function used to construct the code.

        :code:`code_block_size` : :obj:`int`, optional
            The code block size :math:`n`. Must satisfy :math:`2^n \geq |\\mathcal{X}|`, where :math:`|\\mathcal{X}|` is the cardinality of the source alphabet, given by :code:`len(pmf)`.
        """
        self._pmf = np.array(pmf)

        if 2**code_block_size < len(pmf):
            raise ValueError("Code block size is too low")

        super().__init__(sourcewords=TunstallCode._tunstall_algorithm(pmf, code_block_size))

    @property
    def pmf(self):
        """
        The probability mass function used to construct the code. This property is read-only.
        """
        return self._pmf

    @staticmethod
    def _tunstall_algorithm(pmf, code_block_size):
        class Node:
            def __init__(self, symbols, probability):
                self.symbols = symbols
                self.probability = probability
            def __lt__(self, other):
                return -self.probability < -other.probability

        queue = [Node((symbol,), probability) for (symbol, probability) in enumerate(pmf)]
        heapq.heapify(queue)

        while len(queue) + len(pmf) - 1 < 2**code_block_size:
            node = heapq.heappop(queue)
            for (symbol, probability) in enumerate(pmf):
                new_node = Node(node.symbols + (symbol,), node.probability * probability)
                heapq.heappush(queue, new_node)
        sourcewords = sorted(node.symbols for node in queue)

        return sourcewords

    def __repr__(self):
        args = 'pmf={}, code_block_size={}'.format(self._pmf.tolist(), self._code_block_size)
        return '{}({})'.format(self.__class__.__name__, args)


def _parse_prefix_free(input_sequence, dictionary):
    output_sequence = []
    i = 0
    while i < len(input_sequence):
        j = 1
        while i + j <= len(input_sequence):
            try:
                key = tuple(input_sequence[i : i + j])
                output_sequence.extend(dictionary[key])
                break
            except KeyError:
                j += 1
        i += j
    return output_sequence
