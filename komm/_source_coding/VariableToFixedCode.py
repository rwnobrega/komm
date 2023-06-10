import itertools as it

import numpy as np

from .util import _parse_prefix_free


class VariableToFixedCode:
    r"""
    Binary, prefix-free, variable-to-fixed length code. Let $\mathcal{X} = \\{0, 1, \ldots, |\mathcal{X} - 1| \\}$ be the alphabet of some discrete source. A *binary variable-to-fixed length code* of *code block size* $n$ is defined by a (possibly partial) decoding mapping $\mathrm{Dec} : \\{ 0, 1 \\}^n \to \mathcal{X}^+$, where $\mathcal{X}^+$ denotes the set of all finite-length, non-empty strings from the source alphabet. The elements in the image of $\mathrm{Dec}$ are called *sourcewords*.

    Warning:

        Only *prefix-free* codes are considered, in which no sourceword is a prefix of any other sourceword.
    """

    def __init__(self, sourcewords):
        r"""
        Constructor for the class.

        Parameters:

            sourcewords (:obj:`list` of :obj:`tuple` of :obj:`int`): The sourcewords of the code. Must be a list of length at most $2^n$ containing tuples of integers in $\mathcal{X}$. The tuple in position $i$ of :code:`sourcewords` should be equal to $\mathrm{Dec}(v)$, where $v$ is the $i$-th element in the lexicographic ordering of $\\{ 0, 1 \\}^n$.

        Note:

            The code block size $n$ and the source cardinality $|\mathcal{X}|$ are inferred from :code:`sourcewords`.

        Examples:

            >>> code = komm.VariableToFixedCode(sourcewords=[(1,), (2,), (0,1), (0,2), (0,0,0), (0,0,1), (0,0,2)])
            >>> (code.source_cardinality, code.code_block_size)
            (3, 3)
            >>> pprint(code.dec_mapping)
            {(0, 0, 0): (1,),
             (0, 0, 1): (2,),
             (0, 1, 0): (0, 1),
             (0, 1, 1): (0, 2),
             (1, 0, 0): (0, 0, 0),
             (1, 0, 1): (0, 0, 1),
             (1, 1, 0): (0, 0, 2)}
            >>> pprint(code.enc_mapping)
            {(0, 0, 0): (1, 0, 0),
             (0, 0, 1): (1, 0, 1),
             (0, 0, 2): (1, 1, 0),
             (0, 1): (0, 1, 0),
             (0, 2): (0, 1, 1),
             (1,): (0, 0, 0),
             (2,): (0, 0, 1)}
        """
        # TODO: Assert prefix-free
        self._sourcewords = sourcewords
        self._source_cardinality = max(it.chain(*sourcewords)) + 1
        self._code_block_size = (len(sourcewords) - 1).bit_length()
        self._enc_mapping = {}
        self._dec_mapping = {}
        for symbols, bits in zip(it.product(range(2), repeat=self._code_block_size), sourcewords):
            self._enc_mapping[bits] = tuple(symbols)
            self._dec_mapping[tuple(symbols)] = bits

    @property
    def source_cardinality(self):
        r"""
        The cardinality $|\mathcal{X}|$ of the source alphabet.
        """
        return self._source_cardinality

    @property
    def code_block_size(self):
        r"""
        The code block size $n$.
        """
        return self._code_block_size

    @property
    def enc_mapping(self):
        r"""
        The encoding mapping $\mathrm{Enc}$ of the code.
        """
        return self._enc_mapping

    @property
    def dec_mapping(self):
        r"""
        The decoding mapping $\mathrm{Dec}$ of the code.
        """
        return self._dec_mapping

    def rate(self, pmf):
        r"""
        Computes the expected rate $R$ of the code, assuming a given :term:`pmf`. This quantity is given by

        .. math::
           R = \frac{n}{\bar{k}},

        where $n$ is the code block size, and $\bar{k}$ is the expected sourceword length, assuming :term:`i.i.d.` source symbols drawn from $p_X$. It is measured in bits per source symbol.

        Parameters:

            pmf (1D-array of :obj:`float`): The (first-order) probability mass function $p_X$ to be assumed.

        Returns:

            rate (:obj:`float`): The expected rate $R$ of the code.

        Examples:

            >>> code = komm.VariableToFixedCode([(0,0,0), (0,0,1), (0,1), (1,)])
            >>> code.rate([2/3, 1/3])
            0.9473684210526315
        """
        probabilities = np.array([np.prod([pmf[x] for x in symbols]) for symbols in self._sourcewords])
        lengths = [len(symbols) for symbols in self._sourcewords]
        return self._code_block_size / np.dot(lengths, probabilities)

    def encode(self, symbol_sequence):
        r"""
        Encodes a sequence of symbols to its corresponding sequence of bits.

        Parameters:

            symbol_sequence (1D-array of :obj:`int`): The sequence of symbols to be encoded. Must be a 1D-array with elements in $\mathcal{X} = \\{0, 1, \ldots, |\mathcal{X} - 1| \\}$.

        Returns:

            bit_sequence (1D-array of :obj:`int`): The sequence of bits corresponding to :code:`symbol_sequence`.

        Examples:

            >>> code = komm.VariableToFixedCode([(0,0,0), (0,0,1), (0,1), (1,)])
            >>> code.encode([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
            array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
        """
        return np.array(_parse_prefix_free(symbol_sequence, self._enc_mapping))

    def decode(self, bit_sequence):
        r"""
        Decodes a sequence of bits to its corresponding sequence of symbols.

        Parameters:

            bit_sequence (1D-array of :obj:`int`): The sequence of bits to be decoded. Must be a 1D-array with elements in $\\{ 0, 1 \\}$.  Its length must be a multiple of $n$.

        Returns:

            symbol_sequence (1D-array of :obj:`int`): The sequence of symbols corresponding to :code:`bits`.

        Examples:

            >>> code = komm.VariableToFixedCode([(0,0,0), (0,0,1), (0,1), (1,)])
            >>> code.decode([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
            array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        """
        bits_reshaped = np.reshape(bit_sequence, newshape=(-1, self._code_block_size))
        return np.concatenate([self._dec_mapping[tuple(bits)] for bits in bits_reshaped])

    def __repr__(self):
        args = "sourcewords={}".format(self._sourcewords)
        return "{}({})".format(self.__class__.__name__, args)
