import functools
import itertools
import operator

import numpy as np

from .algebra import \
    null_matrix, right_inverse, \
    BinaryPolynomial, BinaryFiniteExtensionField

from .util import \
    binlist2int, int2binlist, binary_iterator, binary_iterator_weight, \
    tag, rst_table

__all__ = ['BlockCode', 'HammingCode', 'SimplexCode', 'GolayCode',
           'RepetitionCode', 'SingleParityCheckCode', 'ReedMullerCode',
           'CyclicCode', 'BCHCode']


class BlockCode:
    """
    General binary linear block code. A *binary linear block code* is a :math:`k`-dimensional subspace of the vector space :math:`\\mathbb{F}_2^n`.  The parameters :math:`n` and :math:`k` are called the code *length* and *dimension* of the code, respectively. The parameter :math:`m = n - k` is called the *redundancy* of the code.

    References: :cite:`Lin.Costello.04` (Ch. 3)

    **Decoding methods**

    [[0]]

    Examples
    ========

    >>> code = komm.BlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> (code.length, code.dimension, code.minimum_distance)
    (6, 3, 3)
    >>> code.generator_matrix
    array([[1, 0, 0, 0, 1, 1],
           [0, 1, 0, 1, 0, 1],
           [0, 0, 1, 1, 1, 0]])
    >>> code.parity_check_matrix
    array([[0, 1, 1, 1, 0, 0],
           [1, 0, 1, 0, 1, 0],
           [1, 1, 0, 0, 0, 1]])
    >>> code.codeword_table()
    array([[0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1, 1],
           [0, 1, 0, 1, 0, 1],
           [1, 1, 0, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [1, 0, 1, 1, 0, 1],
           [0, 1, 1, 0, 1, 1],
           [1, 1, 1, 0, 0, 0]])
    >>> code.codeword_weight_distribution()
    array([1, 0, 0, 4, 3, 0, 0])
    >>> code.coset_leader_table()
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0],
           [1, 0, 0, 1, 0, 0]])
    >>> code.coset_leader_weight_distribution()
    array([1, 6, 1, 0, 0, 0, 0])
    >>> (code.packing_radius, code.covering_radius)
    (1, 2)
    """
    def __init__(self, **kwargs):
        """
        Constructor for the class. It expects one of the following formats:

        **Via generator matrix**

        `komm.BlockCode(generator_matrix=generator_matrix)`

            :code:`generator_matrix` : 2D array of :obj:`int`
                Generator matrix :math:`G` for the code, which is a :math:`k \\times n` binary matrix.

        **Via parity-check matrix**

        `komm.BlockCode(parity_check_matrix=parity_check_matrix)`

            :code:`parity_check_matrix` : 2D array of :obj:`int`
                Parity-check matrix :math:`H` for the code, which is a :math:`m \\times n` binary matrix.

        **Via parity submatrix and information set**

        `komm.BlockCode(parity_submatrix=parity_submatrix, information_set=information_set)`

            :code:`parity_submatrix` : 2D array of :obj:`int`
                Parity submatrix :math:`P` for the code, which is a :math:`k \\times m` binary matrix.

            :code:`information_set` : (1D array of :obj:`int`) or :obj:`str`, optional
                Either an array containing the indices of the information positions, which must be a :math:`k`-sublist of :math:`[0 : n)`, or one of the strings :code:`'left'` or :code:`'right'`. The default value is :code:`'left'`.
        """

        if 'generator_matrix' in kwargs:
            self._init_from_generator_matrix(**kwargs)
        elif 'parity_check_matrix' in kwargs:
            self._init_from_parity_check_matrix(**kwargs)
        elif 'parity_submatrix' in kwargs:
            self._init_from_parity_submatrix(**kwargs)
        else:
            raise ValueError("Either specify 'generator_matrix' or 'parity_check_matrix'" \
                             "or 'parity_submatrix')")

    def _init_from_generator_matrix(self, generator_matrix):
        self._generator_matrix = np.array(generator_matrix, dtype=np.int) % 2
        self._parity_check_matrix = null_matrix(self._generator_matrix)
        self._dimension, self._length = self._generator_matrix.shape
        self._redundancy = self._length - self._dimension
        self._is_systematic = False
        self._constructed_from = 'generator_matrix'

    def _init_from_parity_check_matrix(self, parity_check_matrix):
        self._parity_check_matrix = np.array(parity_check_matrix, dtype=np.int) % 2
        self._generator_matrix = null_matrix(self._parity_check_matrix)
        self._redundancy, self._length = self._parity_check_matrix.shape
        self._dimension = self._length - self._redundancy
        self._is_systematic = False
        self._constructed_from = 'parity_check_matrix'

    def _init_from_parity_submatrix(self, parity_submatrix, information_set='left'):
        self._parity_submatrix = np.array(parity_submatrix, dtype=np.int) % 2
        self._dimension, self._redundancy = self._parity_submatrix.shape
        self._length = self._dimension + self._redundancy
        if information_set == 'left':
            information_set = np.arange(self._dimension)
        elif information_set == 'right':
            information_set = np.arange(self._redundancy, self._length)
        self._information_set = np.array(information_set, dtype=np.int)
        if self._information_set.size != self._dimension or \
           self._information_set.min() < 0 or self._information_set.max() > self._length:
            raise ValueError("Parameter 'information_set' must be a 'k'-subset of 'range(n)'")
        self._parity_set = np.setdiff1d(np.arange(self._length), self._information_set)
        self._generator_matrix = np.empty((self._dimension, self._length), dtype=np.int)
        self._generator_matrix[:, self._information_set] = np.eye(self._dimension, dtype=np.int)
        self._generator_matrix[:, self._parity_set] = self._parity_submatrix
        self._parity_check_matrix = np.empty((self._redundancy, self._length), dtype=np.int)
        self._parity_check_matrix[:, self._information_set] = self._parity_submatrix.T
        self._parity_check_matrix[:, self._parity_set] = np.eye(self._redundancy, dtype=np.int)
        self._is_systematic = True
        self._constructed_from = 'parity_submatrix'

    def __repr__(self):
        if self._constructed_from == 'generator_matrix':
            args = 'generator_matrix={}'.format(self._generator_matrix.tolist())
        elif self._constructed_from == 'parity_check_matrix':
            args = 'parity_check_matrix={}'.format(self._parity_check_matrix.tolist())
        elif self._constructed_from == 'parity_submatrix':
            args = 'parity_submatrix={}, information_set={}'.format(self._parity_submatrix.tolist(), self._information_set.tolist())
        return '{}({})'.format(self.__class__.__name__, args)

    @classmethod
    def _process_docstring(cls):
        cls.__doc__ = cls.__doc__.replace('[[0]]', rst_table(cls._available_decoding_methods()))

    @property
    def length(self):
        """The length :math:`n` of the code. This property is read-only."""
        return self._length

    @property
    def dimension(self):
        """The dimension :math:`k` of the code. This property is read-only."""
        return self._dimension

    @property
    def redundancy(self):
        """The redundancy :math:`m` of the code. This property is read-only."""
        return self._redundancy

    @property
    def minimum_distance(self):
        """The minimum distance :math:`d_\\mathrm{min}` of the code. This is equal to the minimum Hamming weight of the non-zero codewords. This property is read-only."""
        if not hasattr(self, '_minimum_distance'):
            codeword_weight_distribution = self.codeword_weight_distribution()
            self._minimum_distance = np.flatnonzero(codeword_weight_distribution)[1]  # TODO: optimize me
        return self._minimum_distance

    @property
    def packing_radius(self):
        """The packing radius of the code. This is also called the *error-correcting capability* of the code, and is equal to :math:`\\lfloor (d_\\mathrm{min} - 1) / 2 \\rfloor`. This property is read-only."""
        if not hasattr(self, '_packing_radius'):
            self._packing_radius = self.minimum_distance // 2
        return self._packing_radius

    @property
    def covering_radius(self):
        """The covering radius of the code. This is equal to the maximum Hamming weight of the coset leaders. This property is read-only."""
        if not hasattr(self, '_covering_radius'):
            coset_leader_weight_distribution = self.coset_leader_weight_distribution()
            self._covering_radius = np.flatnonzero(coset_leader_weight_distribution)[-1]
        return self._covering_radius

    @property
    def generator_matrix(self):
        """The generator matrix :math:`G` of the code. This property is read-only."""
        return self._generator_matrix

    @property
    def parity_check_matrix(self):
        """The parity-check matrix :math:`H` of the code. This property is read-only."""
        return self._parity_check_matrix

    def codeword_table(self):
        """
        Returns a matrix containing all the codewords.

        **Output:**
            :code:`codeword_table` : 2D array of :obj:`int`
                A :math:`2^k \\times n` matrix whose rows are all the codewords. The codeword in row :math:`i` corresponds to the message whose binary representation (:term:`MSB` in the right) is :math:`i`.

        This is a cached method.

        Examples
        ========
        >>> code = komm.BlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> code.codeword_table()
        array([[0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 1],
               [0, 1, 0, 1, 0, 1],
               [1, 1, 0, 1, 1, 0],
               [0, 0, 1, 1, 1, 0],
               [1, 0, 1, 1, 0, 1],
               [0, 1, 1, 0, 1, 1],
               [1, 1, 1, 0, 0, 0]])
        """
        if not hasattr(self, '_codeword_table'):
            self._codeword_table = np.empty([2**self._dimension, self._length], dtype=np.int)
            for (i, message) in enumerate(binary_iterator(self._dimension)):
                self._codeword_table[i] = self.encode(message)
        return self._codeword_table

    def codeword_weight_distribution(self):
        """
        Returns the codeword weight distribution.

        **Output:**
            :code:`codeword_weight_distribution` : 1D array of :obj:`int`
                An array of shape :math:`(n + 1)` in which element in position :math:`w` is equal to the number of codewords of Hamming weight :math:`w`, for :math:`w \in [0 : n)`.

        This is a cached method.

        Examples
        ========
        >>> code = komm.BlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> code.codeword_weight_distribution()
        array([1, 0, 0, 4, 3, 0, 0])
        """
        if not hasattr(self, '_codeword_weight_distribution'):
            self._codeword_weight_distribution = np.bincount(np.sum(self.codeword_table(), axis=1),
                                                             minlength=self._length + 1)
        return self._codeword_weight_distribution

    def coset_leader_table(self):
        """
        Returns a matrix containing all the coset leaders. This may be used as a :term:`LUT` for syndrome-based decoding.

        **Output:**
            :code:`coset_leader_table` : 2D array of :obj:`int`
                A :math:`2^m \\times n` matrix whose rows are all the coset leaders. The coset leader in row :math:`i` corresponds to the syndrome whose binary representation (:term:`MSB` in the right) is :math:`i`.

        This is a cached method.

        Examples
        ========
        >>> code = komm.BlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> code.coset_leader_table()
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1],
               [0, 1, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0],
               [1, 0, 0, 1, 0, 0]])
        """
        if not hasattr(self, '_coset_leader_table'):
            self._coset_leader_table = np.empty([2**self._redundancy, self._length], dtype=np.int)
            taken = []
            for w in range(self._length + 1):
                for errorword in binary_iterator_weight(self._length, w):
                    syndrome = np.dot(errorword, self._parity_check_matrix.T) % 2
                    syndrome_int = binlist2int(syndrome)
                    if syndrome_int not in taken:
                        self._coset_leader_table[syndrome_int] = np.array(errorword)
                        taken.append(syndrome_int)
                    if len(taken) == 2**self.redundancy:
                        break
        return self._coset_leader_table

    def coset_leader_weight_distribution(self):
        """
        Returns the coset leader weight distribution.

        **Output:**
            :code:`coset_leader_weight_distribution` : 1D array of :obj:`int`
                An array of shape :math:`(n + 1)` in which element in position :math:`w` is equal to the number of coset leaders of weight :math:`w`, for :math:`w \in [0 : n)`.

        This is a cached method.

        >>> code = komm.BlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> code.coset_leader_weight_distribution()
        array([1, 6, 1, 0, 0, 0, 0])
        """
        if not hasattr(self, '_coset_leader_weight_distribution'):
            coset_leader_table = self.coset_leader_table()
            self._coset_leader_weight_distribution = np.bincount([np.count_nonzero(s) for s in coset_leader_table], minlength=self._length + 1)
        return self._coset_leader_weight_distribution

    @property
    @functools.lru_cache(maxsize=None)
    def _generator_matrix_right_inverse(self):
        return right_inverse(self.generator_matrix)

    def encode(self, message, method=None):
        """
        Encode a given message to its corresponding codeword.

        **Input:**
            :code:`message` : 1D array of :obj:`int`
                The message to be encoded. Its length must be :math:`k`.

            :code:`method` : :obj:`str`, optional
                The encoding method to be used.

        **Output:**
            :code:`codeword` : 1D array of :obj:`int`
                The codeword corresponding to :code:`message`. Its length is equal to :math:`n`.
        """
        message = np.array(message)

        if method is None:
            method = self._default_encoder()

        encoder = getattr(self, '_encode_' + method)
        codeword = encoder(message)

        return codeword

    def _encode_generator_matrix(self, message):
        codeword = np.dot(message, self._generator_matrix) % 2
        return codeword

    def _encode_systematic_generator_matrix(self, message):
        codeword = np.empty(self._length, dtype=np.int)
        codeword[self._information_set] = message
        codeword[self._parity_set] = np.dot(message, self._parity_submatrix) % 2
        return codeword

    def _default_encoder(self):
        if self._is_systematic:
            return 'systematic_generator_matrix'
        else:
            return 'generator_matrix'

    def message_from_codeword(self, codeword):
        """
        Returns the message corresponding to a given codeword. In other words, applies the inverse encoding map.

        **Input:**
            :code:`codeword` : 1D array of :obj:`int`
                A codeword from the code. Its length must be :math:`n`.

        **Output:**
            :code:`message` : 1D array of :obj:`int`
                The message corresponding to :code:`codeword`. Its length is equal to :math:`k`.
        """
        if self._is_systematic:
            return codeword[self._information_set]
        else:
            return np.dot(codeword, self._generator_matrix_right_inverse) % 2

    def decode(self, recvword, method=None):
        """
        Decode a received word to a message.

        **Input:**
            :code:`recvword` : 1D array of (:obj:`int` or :obj:`float`)
                The word to be decoded. If using a hard-decision decoding method, then the elements of the array must be bits (integers in :math:`\{ 0, 1 \}`). If using a soft-decision decoding method, then the elements of the array must be soft-bits (floats standing for log-probability ratios, in which positive values represent bit :math:`0` and negative values represent bit :math:`1`). Its length must be :math:`n`.

            :code:`method` : :obj:`str`, optional
                The decoding method to be used.

        **Output:**
            :code:`message_hat` : 1D array of :obj:`int`
                The message decoded from :code:`recvword`. Its length is equal to :math:`k`.
        """
        recvword = np.array(recvword)

        if method is None:
            method = self._default_decoder(recvword.dtype)

        decoder = getattr(self, '_decode_' + method)

        if decoder.target == 'codeword':
            message_hat = self.message_from_codeword(decoder(recvword))
        elif decoder.target == 'message':
            message_hat = decoder(recvword)

        return message_hat

    @tag(name='Exhaustive search (hard-decision)', input_type='hard', target='codeword')
    def _decode_exhaustive_search_hard(self, recvword):
        """
        Exhaustive search minimum distance hard decoder. Hamming distance.
        """
        codewords = self.codeword_table()
        metrics = np.count_nonzero(recvword != codewords, axis=1)
        codeword_hat = codewords[np.argmin(metrics)]
        return codeword_hat

    @tag(name='Exhaustive search (soft-decision)', input_type='soft', target='codeword')
    def _decode_exhaustive_search_soft(self, recvword):
        """
        Exhaustive search minimum distance soft decoder. Euclidean distance.
        """
        codewords = self.codeword_table()
        metrics = np.dot(recvword, codewords.T)
        codeword_hat = codewords[np.argmin(metrics)]
        return codeword_hat

    @tag(name='Syndrome table', input_type='hard', target='codeword')
    def _decode_syndrome_table(self, recvword):
        """
        Syndrome table decoder.
        """
        coset_leader_table = self.coset_leader_table()
        syndrome = np.dot(recvword, self._parity_check_matrix.T) % 2
        syndrome_int = binlist2int(syndrome)
        errorword_hat = coset_leader_table[syndrome_int]
        codeword_hat = np.bitwise_xor(recvword, errorword_hat)
        return codeword_hat

    def _default_decoder(self, dtype):
        if dtype == np.int:
            if self._dimension >= self._redundancy:
                return 'syndrome_table'
            else:
                return 'exhaustive_search_hard'
        elif dtype == np.float:
            return 'exhaustive_search_soft'

    @classmethod
    def _available_decoding_methods(cls):
        header = ['Method', 'Identifier', 'Input type']
        table = [header]
        for name in dir(cls):
            if name.startswith('_decode_'):
                identifier = name[8:]
                method = getattr(cls, name)
                table.append([method.name, ':code:`{}`'.format(identifier), method.input_type])
        return table


class HammingCode(BlockCode):
    """
    Hamming code.

    The *Hamming code* :cite:`Lin.Costello.04` (Sec 4.1) with redundancy :math:`m` is defined as the linear block code with parity-check matrix whose columns are all the ...

    - Length: :math:`n = 2^m - 1`
    - Redundancy: :math:`m`
    - Dimension: :math:`k = 2^m - m - 1`
    - Minimum distance: :math:`d = 3`

    .. rubric:: Decoding methods

    [[0]]

    .. rubric:: Parameters

    m : :obj:`int`
        The redundancy :math:`m` of the code. Must satisfy :math:`m \geq 2`.

    .. rubric:: Notes

    - For :math:`m = 2` it reduces to the repetition code
      (:class:`RepetitionCode`) of length :math:`3`.
    - Its dual is the simplex code (:class:`SimplexCode`).
    - Hamming codes are perfect codes.

    .. rubric:: Examples

    >>> code = komm.HammingCode(3)
    >>> (code.length, code.dimension, code.minimum_distance)
    (7, 4, 3)
    >>> code.generator_matrix
    array([[1, 0, 0, 0, 1, 1, 0],
           [0, 1, 0, 0, 1, 0, 1],
           [0, 0, 1, 0, 0, 1, 1],
           [0, 0, 0, 1, 1, 1, 1]])
    >>> code.parity_check_matrix
    array([[1, 1, 0, 1, 1, 0, 0],
           [1, 0, 1, 1, 0, 1, 0],
           [0, 1, 1, 1, 0, 0, 1]])
    >>> code.encode([1, 0, 1, 1])
    array([0, 1, 0, 1, 0, 1, 1])
    >>> code.decode([0, 1, 0, 0, 0, 1, 1])
    array([1, 0, 1, 1])

    ..rubric:: See also

    BlockCode, SimplexCode, GolayCode, RepetitionCode
    """
    def __init__(self, m):
        super().__init__(parity_submatrix=HammingCode._hamming_parity_submatrix(m))
        self._minimum_distance = 3

    def __repr__(self):
        args = 'redundancy={}'.format(self._redundancy)
        return '{}({})'.format(self.__class__.__name__, args)

    @staticmethod
    def _hamming_parity_submatrix(m):
        P_list = []
        for w in range(2, m + 1):
            for row in binary_iterator_weight(m, w):
                P_list.append(row)
        return np.array(P_list, dtype=np.int)


class SimplexCode(BlockCode):
    """
    Simplex (maximum-length) code.

    Simplex code (also known as maximum-length codes) with dimension :math:`k`.

    - Length: :math:`n = 2^k - 1`
    - Dimension: :math:`k`
    - Redundancy: :math:`m = 2^k - k - 1`
    - Minimum distance: :math:`d = 2^{k - 1}`

    .. rubric:: Decoding methods

    [[0]]

    .. rubric:: Parameters

    k : :obj:`int`
        The dimension :math:`k` of the code. Must satisfy :math:`m \geq 2`.

    .. rubric:: Notes

    - For :math:`k = 2` it reduces to the single parity check code
      (:class:`SingleParityCheckCode`) of length :math:`3`.
    - Its dual is the Hamming code (:class:`HammingCode`).
    - Simplex codes are constant-weight codes.

    .. rubric:: Examples

    >>> code = komm.SimplexCode(3)
    >>> (code.length, code.dimension, code.minimum_distance)
    (7, 3, 4)
    >>> code.generator_matrix
    array([[1, 0, 0, 1, 1, 0, 1],
           [0, 1, 0, 1, 0, 1, 1],
           [0, 0, 1, 0, 1, 1, 1]])
    >>> code.parity_check_matrix
    array([[1, 1, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 0],
           [0, 1, 1, 0, 0, 1, 0],
           [1, 1, 1, 0, 0, 0, 1]])
    >>> code.encode([1, 0, 1])
    array([1, 0, 1, 1, 0, 1, 0])
    >>> code.decode([1, 0, 1, 1, 1, 1, 0])
    array([1, 0, 1])

    .. rubric:: See also

    :class:`BlockCode`, :class:`HammingCode`, :class:`GolayCode`, :class:`RepetitionCode`
    """
    def __init__(self, k):
        super().__init__(parity_submatrix=HammingCode._hamming_parity_submatrix(k).T)
        self._minimum_distance = 2**(k - 1)

    def __repr__(self):
        args = 'dimension={}'.format(self._dimension)
        return '{}({})'.format(self.__class__.__name__, args)


class GolayCode(BlockCode):
    """
    Binary Golay code.

    - Length: 23
    - Dimension: 12
    - Minimum distance: 7

    .. rubric:: Decoding methods

    [[0]]

    .. rubric:: Notes

    - The binary Golay code is a perfect code.

    .. rubric:: Examples

    >>> code = komm.GolayCode()
    >>> (code.length, code.dimension, code.minimum_distance)
    (23, 12, 7)
    >>> recvword = np.zeros(23, dtype=np.int); recvword[[2, 10, 19]] = 1
    >>> code.decode(recvword)  # Golay code can correct up to 3 errors.
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> recvword = np.zeros(23, dtype=np.int); recvword[[2, 3, 10, 19]] = 1
    >>> code.decode(recvword)  # Golay code cannot correct more than 3 errors.
    array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    """
    def __init__(self):
        super().__init__(parity_submatrix=GolayCode._golay_parity_submatrix())
        self._minimum_distance = 7

    def __repr__(self):
        args = ''
        return '{}({})'.format(self.__class__.__name__, args)

    @staticmethod
    def _golay_parity_submatrix():
        return np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
             [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
             [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
             [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
             [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
             [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
             [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
             [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
             [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
             [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1]])


class RepetitionCode(BlockCode):
    """
    Repetition code.

    Repetition code of length :math:`n`.

    - Length: :math:`n`
    - Dimension: :math:`k = 1`
    - Minimum distance: :math:`d = n`

    .. rubric:: Decoding methods

    [[0]]

    .. rubric:: Parameters

    n : :obj:`int`
        The length :math:`n` of the code. Must be a positive integer.

    .. rubric:: Notes

    - Its dual is the single parity check code (:class:`SingleParityCheckCode`).

    .. rubric:: Examples

    >>> code = komm.RepetitionCode(5)
    >>> (code.length, code.dimension, code.minimum_distance)
    (5, 1, 5)
    >>> code.generator_matrix
    array([[1, 1, 1, 1, 1]])
    >>> code.parity_check_matrix
    array([[1, 1, 0, 0, 0],
           [1, 0, 1, 0, 0],
           [1, 0, 0, 1, 0],
           [1, 0, 0, 0, 1]])
    >>> code.encode([1])
    array([1, 1, 1, 1, 1])
    >>> code.decode([1, 0, 1, 0, 0])
    array([0])
    """
    def __init__(self, n):
        super().__init__(parity_submatrix=np.ones((1, n - 1), dtype=np.int))
        self._minimum_distance = n

    def __repr__(self):
        args = 'length={}'.format(self._length)
        return '{}({})'.format(self.__class__.__name__, args)

    @tag(name='Majority-logic', input_type='hard', target='codeword')
    def _decode_majority_logic(self, recvword):
        """
        Majority-logic decoder.

        A hard-decision decoder for Repetition codes only.
        """
        majority = np.argmax(np.bincount(recvword))
        codeword_hat = majority * np.ones_like(recvword)
        return codeword_hat

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'majority_logic'
        else:
            return super()._default_decoder(dtype)


class SingleParityCheckCode(BlockCode):
    """
    Single parity check code.

    Single parity check code of length :math:`n`.

    - Length: :math:`n`.
    - Dimension: :math:`k = n - 1`.
    - Minimum distance: :math:`d = 2`.

    .. rubric:: Decoding methods

    [[0]]

    .. rubric:: Parameters

    n : :obj:`int`
        The length :math:`n` of the code. Must be a positive integer.

    .. rubric:: Notes

    - Its dual is the repetition code (:class:`RepetitionCode`).

    .. rubric:: Examples

    >>> code = komm.SingleParityCheckCode(5)
    >>> (code.length, code.dimension, code.minimum_distance)
    (5, 4, 2)
    >>> code.generator_matrix
    array([[1, 0, 0, 0, 1],
           [0, 1, 0, 0, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 0, 1, 1]])
    >>> code.parity_check_matrix
    array([[1, 1, 1, 1, 1]])
    >>> code.encode([1, 0, 1, 1])
    array([1, 0, 1, 1, 1])
    """
    def __init__(self, n):
        super().__init__(parity_submatrix=np.ones((1, n - 1), dtype=np.int).T)
        self._minimum_distance = 2

    def __repr__(self):
        args = 'length={}'.format(self._length)
        return '{}({})'.format(self.__class__.__name__, args)

    @tag(name='Wagner', input_type='soft', target='codeword')
    def _decode_wagner(self, recvword):
        """
        Wagner decoder.

        A soft-decision decoder for SingleParityCheck codes only.

        References
        ----------
        [1] Costello, Forney: Channel Coding: The Road to Channel Capacity.
        """
        codeword_hat = (recvword < 0)
        if np.bitwise_xor.reduce(codeword_hat) != 0:
            i = np.argmin(np.abs(recvword))
            codeword_hat[i] ^= 1
        return codeword_hat.astype(np.int)

    def _default_decoder(self, dtype):
        if dtype == np.float:
            return 'wagner'
        else:
            return super()._default_decoder(dtype)


class ReedMullerCode(BlockCode):
    """
    Reed--Muller code.

    Reed--Muller code with parameters :math:`r` and :math:`m`.

    - Length: :math:`n = 2^m`
    - Dimension: :math:`k = 1 + {m \choose 1} + \cdots + {m \choose r}`
    - Minimum distance: :math:`d = 2^{m - r}`

    The parameters must satisfy :math:`0 \leq r < m`.

    .. rubric:: Decoding methods

    [[0]]

    .. rubric:: Parameters

    r : :obj:`int`
        The parameter :math:`r` of the code.
    m : :obj:`int`
        The parameter :math:`m` of the code.

    .. rubric:: Notes

    - For :math:`r = 0` it reduces to a repetition code (:class:`RepetitionCode`).
    - For :math:`r = 1` it reduces to an extended simplex code (:class:`SimplexCode`).
    - For :math:`r = m-2` it reduces to an extended Hamming code (:class:`HammingCode`).
    - For :math:`r = m-1` it reduces to a single parity check code (:class:`SingleParityCheckCode`).

    .. rubric:: Examples

    >>> code = komm.ReedMullerCode(1, 5)
    >>> (code.length, code.dimension, code.minimum_distance)
    (32, 6, 16)

    .. rubric:: See also

    BlockCode, HammingCode, SimplexCode, RepetitionCode, SingleParityCheckCode
    """
    def __init__(self, r, m):
        assert 0 <= r < m

        super().__init__(generator_matrix=ReedMullerCode._reed_muller_generator_matrix(r, m))
        self._minimum_distance = 2**(m - r)
        self._r = r
        self._m = m

    def __repr__(self):
        args = '{}, {}'.format(self._r, self._m)
        return '{}({})'.format(self.__class__.__name__, args)

    @staticmethod
    def _reed_muller_generator_matrix(r, m):
        """
        [1] Lin, Costello, 2Ed, p. 105--114.
        Assumes 0 <= r < m.
        """
        v = np.empty((m, 2**m), dtype=np.int)
        for i in range(m):
            block = np.hstack((np.zeros(2**(m - i - 1), dtype=np.int), np.ones(2**(m - i - 1), dtype=np.int)))
            v[m - i - 1] = np.tile(block, 2**i)

        G_list = []
        for ell in range(r, 0, -1):
            for I in itertools.combinations(range(m), ell):
                row = functools.reduce(np.multiply, v[I, :])
                G_list.append(row)
        G_list.append(np.ones(2**m, dtype=np.int))

        return np.array(G_list, dtype=np.int)

    @functools.lru_cache(maxsize=None)
    def reed_partitions(self):
        """
        Get Reed partitions from Reed-Muller generator matrix.

        References
        ----------
        [1] Lin, Costello, 2Ed, pp. 105--114.
        """
        reed_partitions = []
        for ell in range(self._r, -1, -1):
            binary_vectors_I = np.array(list(binary_iterator(ell)), dtype=np.int)
            binary_vectors_J = np.array(list(binary_iterator(self._m - ell)), dtype=np.int)
            for I in itertools.combinations(range(self._m), ell):
                E = np.setdiff1d(np.arange(self._m), I, assume_unique=True)
                S = np.dot(binary_vectors_I, 2**np.array(I, dtype=np.int))
                Q = np.dot(binary_vectors_J, 2**np.array(E, dtype=np.int))
                reed_partitions.append(S[np.newaxis] + Q[np.newaxis].T)
        return reed_partitions

    @tag(name='Reed', input_type='hard', target='message')
    def _decode_reed(self, recvword):
        """
        Reed decoding algorithm for Reed--Muller codes.
        It's a majority-logic decoding algorithm.

        References
        ----------
        [1] Lin, Costello, 2Ed, pp. 105--114, 439--440.
        """
        reed_partitions = self.reed_partitions()
        message_hat = np.empty(self._generator_matrix.shape[0], dtype=np.int)
        bx = np.copy(recvword)
        for idx, partition in enumerate(reed_partitions):
            checksums = np.bitwise_xor.reduce(bx[partition], axis=1)
            message_hat[idx] = np.count_nonzero(checksums) > len(checksums) // 2
            bx ^= message_hat[idx] * self._generator_matrix[idx]
        return message_hat

    @tag(name='Weighted Reed', input_type='soft', target='message')
    def _decode_weighted_reed(self, recvword):
        """
        References
        ----------
        [1] Lin, Costello, 2Ed, pp. 440-442.
        """
        reed_partitions = self.reed_partitions()
        message_hat = np.empty(self._generator_matrix.shape[0], dtype=np.int)
        bx = (recvword < 0) * 1
        for idx, partition in enumerate(reed_partitions):
            checksums = np.bitwise_xor.reduce(bx[partition], axis=1)
            min_reliability = np.min(np.abs(recvword[partition]), axis=1)
            decision_var = np.dot(1 - 2*checksums, min_reliability)
            message_hat[idx] = decision_var < 0
            bx ^= message_hat[idx] * self._generator_matrix[idx]
        return message_hat

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'reed'
        elif dtype == np.float:
            return 'weighted_reed'


class CyclicCode(BlockCode):
    """
    General binary cyclic code.

    Examples of generator polynomials can be found in the table below.

    ===============    ======================
    Code               Generator polynomial
    ===============    ======================
    Hamming (7,4,3)    :code:`0b1011`
    Simplex (7,3,4)    :code:`0b10111`
    BCH (15,5,7)       :code:`0b10100110111`
    Golay (23,12,7)    :code:`0b101011100011`
    ===============    ======================

    .. rubric:: Decoding methods

    [[0]]

    .. rubric:: Parameters

    length : :obj:`int`
        Length :math:`n` of the code.
    generator_polynomial : :obj:`int`
        Generator polynomial :math:`G(X)` of the code. The polynomial :math:`X^3 + X + 1` is represented as :code:`0b1011 = 0o13 = 11`.
    systematic : :obj:`bool`
        True for systematic encoder (default), False otherwise.

    .. rubric:: Examples

    >>> code = komm.CyclicCode(length=7, generator_polynomial=0b1011)  # Hamming(3)
    >>> (code.length, code.dimension, code.minimum_distance)
    (7, 4, 3)

    >>> code = komm.CyclicCode(length=23, generator_polynomial=0b101011100011)  # Golay()
    >>> (code.length, code.dimension, code.minimum_distance)
    (23, 12, 7)

    .. rubric:: See also

    BlockCode, BCHCode
    """
    def __init__(self, length, generator_polynomial, systematic=True):
        self._generator_polynomial = BinaryPolynomial(generator_polynomial)
        self._length = length
        self._redundancy = self._generator_polynomial.degree
        self._dimension = self._length - self._redundancy
        self._modulus = BinaryPolynomial.from_exponents([1, self._length])
        self._parity_polynomial = self._modulus // self._generator_polynomial
        self._is_systematic = bool(systematic)
        if self._is_systematic:
            self._information_set = np.arange(self._redundancy, self._length)

    def __repr__(self):
        args = '{}, generator_polynomial={}, systematic={}'.format(self._length, self._generator_polynomial, self._is_systematic)
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def generator_polynomial(self):
        """Generator polynomial :math:`g(X)` of the cyclic code, of degree :math:`n - k`."""
        return self._generator_polynomial

    @property
    def parity_polynomial(self):
        """Parity polynomial :math:`h(X)` of the cyclic code, of degree :math:`k`."""
        return self._parity_polynomial

    @functools.lru_cache(maxsize=None)
    def meggitt_table(self):
        """
        Meggitt table for Meggit decoder.

        References
        ==========
        .. [1] Sebastià Xambó i Descamps: A computational primer on BLOCK ERROR-CORRECTING CODES
        """
        meggitt_table = {}
        for w in range(self.packing_radius):
            for idx in itertools.combinations(range(self._length - 1), w):
                errorword_poly = BinaryPolynomial.from_exponents(list(idx) + [self._length - 1])
                syndrome_poly = errorword_poly % self._generator_polynomial
                meggitt_table[syndrome_poly] = errorword_poly
        return meggitt_table

    def _encode_cyclic_direct(self, message):
        """
        Encoder for cyclic codes. Direct, non-systematic method.
        """
        message_poly = BinaryPolynomial.from_coefficients(message)
        return (message_poly * self._generator_polynomial).coefficients(width=self._length)

    def _encode_cyclic_systematic(self, message):
        """
        Encoder for cyclic codes. Systematic method.
        """
        message_poly = BinaryPolynomial.from_coefficients(message)
        message_poly_shifted = message_poly << self._generator_polynomial.degree
        parity = message_poly_shifted % self._generator_polynomial
        return (message_poly_shifted + parity).coefficients(width=self._length)

    def _default_encoder(self):
        if self._is_systematic:
            return 'cyclic_systematic'
        else:
            return 'cyclic_direct'

    @property
    @functools.lru_cache(maxsize=None)
    def generator_matrix(self):
        """Generator matrix :math:`G` of the cyclic code."""
        n, k = self.length, self.dimension
        generator_matrix = np.empty((k, n), dtype=np.int)
        row = self._generator_polynomial.coefficients(width=n)
        for i in range(k):
            generator_matrix[i] = np.roll(row, i)
        return generator_matrix

    @property
    @functools.lru_cache(maxsize=None)
    def parity_check_matrix(self):
        """Parity-check matrix :math:`H` of the cyclic code."""
        n, k = self.length, self.dimension
        parity_check_matrix = np.empty((n - k, n), dtype=np.int)
        row = self._parity_polynomial.coefficients(width=n)[::-1]
        for i in range(n - k):
            parity_check_matrix[n - k - i - 1] = np.roll(row, -i)
        return parity_check_matrix

    @tag(name='Meggitt decoder', input_type='hard', target='codeword')
    def _decode_meggitt(self, recvword):
        """
        Meggitt decoder.

        References
        ==========
        .. [1] Sebastià Xambó i Descamps: A computational primer on BLOCK ERROR-CORRECTING CODES
        """
        meggitt_table = self.meggitt_table()
        recvword_poly = BinaryPolynomial.from_coefficients(recvword)
        syndrome_poly = recvword_poly % self._generator_polynomial
        if syndrome_poly == 0:
            return recvword
        errorword_hat_poly = BinaryPolynomial(0)
        for j in range(self._length):
            if syndrome_poly in meggitt_table:
                errorword_hat_poly = meggitt_table[syndrome_poly] // (1 << j)
                break
            syndrome_poly = (syndrome_poly << 1) % self._generator_polynomial
        return (recvword_poly + errorword_hat_poly).coefficients(self._length)

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'meggitt'
        else:
            return super()._default_decoder(dtype)


class BCHCode(CyclicCode):
    """
    Bose--Chaudhuri--Hocquenghem (BCH) code.

    .. rubric:: Decoding methods

    [[0]]

    .. rubric:: Parameters

    length : :obj:`int`
        Length :math:`n` of the code.
    t : :obj:`int`
        Designed error-correcting capability :math:`t` of the BCH code. It will be internally
        replaced by the true error-correcting capability of the code.

    .. rubric:: Examples

    >>> code = komm.BCHCode(5, 3)
    >>> (code.length, code.dimension, code.minimum_distance)
    (31, 16, 7)
    >>> code.generator_polynomial
    0b1000111110101111

    >>> # Example of t being replaced by its true value.
    >>> # Here.

    .. rubric:: See also

    CyclicCode
    """
    def __init__(self, m, designed_t):
        assert 1 <= designed_t < 2**(m - 1)

        field = BinaryFiniteExtensionField(m)
        generator_polynomial, t = self._bch_code_generator_polynomial(field, m, designed_t)
        super().__init__(length=2**m - 1, generator_polynomial=generator_polynomial)

        self._field = field
        self._m = m
        self._packing_radius = t
        self._minimum_distance = 2*t + 1

        alpha = field.primitive_element
        self._beta = [alpha**(i + 1) for i in range(2*t)]
        self._beta_minimal_polynomial = [b.minimal_polynomial() for b in self._beta]

    def __repr__(self):
        args = '{}, {}'.format(self._m, self._packing_radius)
        return '{}({})'.format(self.__class__.__name__, args)

    @staticmethod
    def _bch_code_generator_polynomial(field, m, designed_t):
        """
        [1] Lin, Costello, 2Ed, p. 194--195.
        Assumes 1 <= designed_t < 2**(m - 1)
        """
        alpha = field.primitive_element

        t = designed_t
        lcm_set = {(alpha**(2*i + 1)).minimal_polynomial() for i in range(t)}
        while True:
            if (alpha**(2*t + 1)).minimal_polynomial() not in lcm_set:
                break
            t += 1
        generator_polynomial = functools.reduce(operator.mul, lcm_set)

        return generator_polynomial, t

    def _bch_general_decoder(self, recvword, syndrome_computer, key_equation_solver, root_finder):
        """
        General BCH decoder.

        References
        ==========
        .. [1] Lin-Costello, p. 205--209.
        """
        recvword_poly = BinaryPolynomial.from_coefficients(recvword)
        syndrome_poly = syndrome_computer(recvword_poly)
        if np.all([x == self._field(0) for x in syndrome_poly]):
            return recvword
        error_location_poly = key_equation_solver(syndrome_poly)
        error_locations = [e.inverse().logarithm() for e in root_finder(error_location_poly)]
        errorword = np.bincount(error_locations, minlength=recvword.size)
        return np.bitwise_xor(recvword, errorword)

    def _bch_syndrome(self, recvword_poly):
        """
        BCH syndrome computation.

        References
        ==========
        .. [1] Lin-Costello, p. 205--209.
        """
        syndrome_poly = np.empty(len(self._beta), dtype=np.object)
        for i, (b, b_min_poly) in enumerate(zip(self._beta, self._beta_minimal_polynomial)):
            syndrome_poly[i] = (recvword_poly % b_min_poly).evaluate(b)
        return syndrome_poly

    def _find_roots(self, polynomial):
        """
        Exhaustive search.
        """
        zero = self._field(0)
        roots = []
        for i in range(self._field.order):
            x = self._field(i)
            evaluated = zero
            for coefficient in reversed(polynomial):  # Horner's method
                evaluated = evaluated * x + coefficient
            if evaluated == zero:
                roots.append(x)
                if len(roots) >= len(polynomial) - 1:
                    break
        return roots

    def _berlekamp_algorithm(self, syndrome_poly):
        """
        Berlekamp's iterative procedure for finding the error-location polynomial of a BCH code.

        References
        ==========
        .. [1] Lin-Costello, p. 209--212
        .. [2] Ryan--Lin, p. 114-121
        """
        field = self._field
        t = self._packing_radius

        sigma = {-1: np.array([field(1)], dtype=np.object), 0: np.array([field(1)], dtype=np.object)}
        discrepancy = {-1: field(1), 0: syndrome_poly[0]}
        degree = {-1: 0, 0: 0}

        for mu in range(2*t):
            if discrepancy[mu] == field(0):
                degree[mu + 1] = degree[mu]
                sigma[mu + 1] = sigma[mu]
            else:
                rho, max_so_far = -1, -1
                for i in range(-1, mu):
                    if discrepancy[i] != field(0) and i - degree[i] > max_so_far:
                        rho, max_so_far = i, i - degree[i]
                degree[mu + 1] = max(degree[mu], degree[rho] + mu - rho)
                sigma[mu + 1] = np.array([field(0)] * (degree[mu + 1] + 1), dtype=np.object)
                first_guy = np.array([field(0)] * (degree[mu + 1] + 1), dtype=np.object)
                first_guy[:degree[mu] + 1] = sigma[mu]
                second_guy = np.array([field(0)] * (degree[mu + 1] + 1), dtype=np.object)
                second_guy[mu-rho : degree[rho] + mu - rho + 1] = sigma[rho]
                e = discrepancy[mu] / discrepancy[rho]
                second_guy = np.array([e * x for x in second_guy], dtype=np.object)
                sigma[mu + 1] = first_guy + second_guy
            if mu < 2*t - 1:
                discrepancy[mu + 1] = syndrome_poly[mu + 1]
                for idx in range(1, degree[mu + 1] + 1):
                    discrepancy[mu + 1] += sigma[mu + 1][idx] * syndrome_poly[mu + 1 - idx]

        return sigma[2*t]

    @tag(name='Berlekamp decoder', input_type='hard', target='codeword')
    def _decode_berlekamp(self, recvword):
        return self._bch_general_decoder(recvword,
                                         syndrome_computer=self._bch_syndrome,
                                         key_equation_solver=self._berlekamp_algorithm,
                                         root_finder=self._find_roots)

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'berlekamp'
        else:
            return super()._default_decoder(dtype)


for cls in __all__:
    eval(cls)._process_docstring()
