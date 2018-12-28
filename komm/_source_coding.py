import heapq

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
        Computes the average length :math:`\\bar{\\ell}` of the symbol code assuming a given :term:`pmf`. It is given by

        .. math::

            \\bar{\\ell} = \\mathrm{E}[\\ell(X)] = \\sum_{x \\in \\mathcal{X}} p_X(x) \\ell(x),

        where :math:`p_X(x)` is the :term:`pmf` to be assumed, and :math:`\\ell(x)` is the number of bits in :math:`\\mathrm{Enc}(x)`.

        **Input:**

        :code:`pmf` : 1D-array of :obj:`float`
            The probability mass function :math:`p_X(x)` to be assumed.

        **Output:**

        :code:`average_length` : :obj:`float`
            The average length :math:`\\bar{\\ell}` of the symbol code assuming the given :term:`pmf`.
        """
        return np.dot([len(bit_sequence) for bit_sequence in self._mapping], pmf) / np.sum(pmf)

    def variance(self, pmf):
        """
        Computes the variance of the length of the symbol code assuming a given :term:`pmf`. It is given by

        .. math::

            \\mathrm{var}[\\ell(X)] = \\mathrm{E}[(\\ell(X) - \\bar{\\ell})^2],

        where :math:`\\ell(x)` is the number of bits in :math:`\\mathrm{Enc}(x)`, and :math:`\\bar{\\ell}` is the average length of the code.

        **Input:**

        :code:`pmf` : 1D-array of :obj:`float`
            The probability mass function :math:`p_X(x)` to be assumed.

        **Output:**

        :code:`variance` : :obj:`float`
            The variance :math:`\\mathrm{var}[\\ell(X)]` of the symbol code assuming the given :term:`pmf`.
        """
        average_length = self.average_length(pmf)
        return np.dot([(len(bit_sequence) - average_length)**2 for bit_sequence in self._mapping], pmf) / np.sum(pmf)

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


class HuffmanCode(PrefixCode):
    """
    Huffman code. It is an optimum prefix code (:class:`PrefixCode`) for a given probability mass function.

    .. rubric:: Examples

    >>> code = komm.HuffmanCode([0.2, 0.4, 0.2, 0.1, 0.1])
    >>> code.mapping
    [(1, 1), (0, 0), (1, 0), (0, 1, 1), (0, 1, 0)]
    >>> code.encode([0, 1, 2, 3, 4])
    array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    """
    def __init__(self, pmf, policy='high'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`pmf` : 1D-array of :obj:`float`
            The probability mass function used to construct the code.

        :code:`policy` : :obj:`str`, optional
            The policy to be used when constructing the code. It must be either :code:`'high'` (move combined symbols as high as possible) or :code:`'low'` (move combined symbols as low as possible). The default value is :code:`'high'`.
        """
        self._pmf = np.array(pmf)
        self._policy = policy

        if policy not in ['high', 'low']:
            raise ValueError("Parameter 'policy' must be in {'high', 'low'}")

        mapping = HuffmanCode._get_mapping(pmf, policy)
        super().__init__(mapping)

    @property
    def pmf(self):
        """
        The probability mass function used to construct the code. This property is read-only.
        """
        return self._pmf

    @staticmethod
    def _get_mapping(pmf, policy):
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

        tree = [Node(i, p) for (i, p) in enumerate(pmf)]
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

        mapping = []
        for symbol in range(len(pmf)):
            node = tree[symbol]
            bits = []
            while node.parent is not None:
                bits.insert(0, node.bit)
                node = tree[node.parent]
            mapping.append(tuple(bits))

        return mapping

    def __repr__(self):
        args = 'pmf={}'.format(self._pmf.tolist())
        return '{}({})'.format(self.__class__.__name__, args)


class TunstallCode:
    """
    Tunstall code [Not implemented yet].
    """
    pass
