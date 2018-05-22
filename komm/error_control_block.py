import functools
import itertools
import operator

import numpy as np

from .algebra import \
    null_matrix, right_inverse, \
    BinaryPolynomial, BinaryFiniteExtensionField

from .util import \
    binlist2int, binary_iterator, binary_iterator_weight, \
    tag, rst_table

__all__ = ['BlockCode', 'HammingCode', 'SimplexCode', 'GolayCode',
           'RepetitionCode', 'SingleParityCheckCode', 'ReedMullerCode',
           'CyclicCode', 'BCHCode']


class BlockCode:
    """
    General binary linear block code. It is characterized by its *generator matrix* :math:`G`, a binary :math:`k \\times n` matrix, and by its *parity-check matrix* :math:`H`, a binary :math:`m \\times n` matrix. Those matrix are related by :math:`G H^\\top = 0`. The parameters :math:`k`, :math:`m`, and :math:`n` are called the code *dimension*, *redundancy*, and *length*, respectively, and are related by :math:`k + m = n`.

    References: :cite:`Lin.Costello.04` (Ch. 3)

    **Decoding methods**

    [[0]]

    .. rubric:: Examples

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

        :code:`generator_matrix` : 2D-array of :obj:`int`
            Generator matrix :math:`G` for the code, which is a :math:`k \\times n` binary matrix.

        **Via parity-check matrix**

        `komm.BlockCode(parity_check_matrix=parity_check_matrix)`

        :code:`parity_check_matrix` : 2D-array of :obj:`int`
            Parity-check matrix :math:`H` for the code, which is an :math:`m \\times n` binary matrix.

        **Via parity submatrix and information set**

        `komm.BlockCode(parity_submatrix=parity_submatrix, information_set=information_set)`

        :code:`parity_submatrix` : 2D-array of :obj:`int`
            Parity submatrix :math:`P` for the code, which is a :math:`k \\times m` binary matrix.

        :code:`information_set` : (1D-array of :obj:`int`) or :obj:`str`, optional
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
        """
        The length :math:`n` of the code. This property is read-only.
        """
        return self._length

    @property
    def dimension(self):
        """
        The dimension :math:`k` of the code. This property is read-only.
        """
        return self._dimension

    @property
    def redundancy(self):
        """
        The redundancy :math:`m` of the code. This property is read-only.
        """
        return self._redundancy

    @property
    def minimum_distance(self):
        """
        The minimum distance :math:`d` of the code. This is equal to the minimum Hamming weight of the non-zero codewords. This property is read-only.
        """
        if not hasattr(self, '_minimum_distance'):
            codeword_weight_distribution = self.codeword_weight_distribution()
            self._minimum_distance = np.flatnonzero(codeword_weight_distribution)[1]  # TODO: optimize me
        return self._minimum_distance

    @property
    def packing_radius(self):
        """
        The packing radius of the code. This is also called the *error-correcting capability* of the code, and is equal to :math:`\\lfloor (d - 1) / 2 \\rfloor`. This property is read-only.
        """
        if not hasattr(self, '_packing_radius'):
            self._packing_radius = self.minimum_distance // 2
        return self._packing_radius

    @property
    def covering_radius(self):
        """
        The covering radius of the code. This is equal to the maximum Hamming weight of the coset leaders. This property is read-only.
        """
        if not hasattr(self, '_covering_radius'):
            coset_leader_weight_distribution = self.coset_leader_weight_distribution()
            self._covering_radius = np.flatnonzero(coset_leader_weight_distribution)[-1]
        return self._covering_radius

    @property
    def generator_matrix(self):
        """
        The generator matrix :math:`G` of the code. It as a :math:`k \\times n` binary matrix, where :math:`k` is the code dimension, and :math:`n` is the code length. This property is read-only.
        """
        return self._generator_matrix

    @property
    def parity_check_matrix(self):
        """
        The parity-check matrix :math:`H` of the code. It as an :math:`m \\times n` binary matrix, where :math:`m` is the code redundancy, and :math:`n` is the code length. This property is read-only.
        """
        return self._parity_check_matrix

    def codeword_table(self):
        """
        Returns a matrix containing all the codewords.

        **Output:**

        :code:`codeword_table` : 2D-array of :obj:`int`
            A :math:`2^k \\times n` matrix whose rows are all the codewords. The codeword in row :math:`i` corresponds to the message whose binary representation (:term:`MSB` in the right) is :math:`i`.

        This is a cached method.

        .. rubric:: Examples

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

        :code:`codeword_weight_distribution` : 1D-array of :obj:`int`
            An array of shape :math:`(n + 1)` in which element in position :math:`w` is equal to the number of codewords of Hamming weight :math:`w`, for :math:`w \\in [0 : n)`.

        This is a cached method.

        .. rubric:: Examples

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

        :code:`coset_leader_table` : 2D-array of :obj:`int`
            A :math:`2^m \\times n` matrix whose rows are all the coset leaders. The coset leader in row :math:`i` corresponds to the syndrome whose binary representation (:term:`MSB` in the right) is :math:`i`.

        This is a cached method.

        .. rubric:: Examples

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

        :code:`coset_leader_weight_distribution` : 1D-array of :obj:`int`
            An array of shape :math:`(n + 1)` in which element in position :math:`w` is equal to the number of coset leaders of weight :math:`w`, for :math:`w \\in [0 : n)`.

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
        Encodes a given message to its corresponding codeword.

        **Input:**

        :code:`message` : 1D-array of :obj:`int`
            The message to be encoded. Its length must be :math:`k`.

        :code:`method` : :obj:`str`, optional
            The encoding method to be used.

        **Output:**

        :code:`codeword` : 1D-array of :obj:`int`
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

        :code:`codeword` : 1D-array of :obj:`int`
            A codeword from the code. Its length must be :math:`n`.

        **Output:**

        :code:`message` : 1D-array of :obj:`int`
            The message corresponding to :code:`codeword`. Its length is equal to :math:`k`.
        """
        if self._is_systematic:
            return codeword[self._information_set]
        else:
            return np.dot(codeword, self._generator_matrix_right_inverse) % 2

    def decode(self, recvword, method=None):
        """
        Decodes a received word to a message.

        **Input:**

        :code:`recvword` : 1D-array of (:obj:`int` or :obj:`float`)
            The word to be decoded. If using a hard-decision decoding method, then the elements of the array must be bits (integers in :math:`\{ 0, 1 \}`). If using a soft-decision decoding method, then the elements of the array must be soft-bits (floats standing for log-probability ratios, in which positive values represent bit :math:`0` and negative values represent bit :math:`1`). Its length must be :math:`n`.

        :code:`method` : :obj:`str`, optional
            The decoding method to be used.

        **Output:**

        :code:`message_hat` : 1D-array of :obj:`int`
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
        header = ['Method', 'Identifier', 'Input\xa0type']
        table = [header]
        for name in dir(cls):
            if name.startswith('_decode_'):
                identifier = name[8:]
                method = getattr(cls, name)
                table.append([method.name, ':code:`{}`'.format(identifier), method.input_type])
        return table


class HammingCode(BlockCode):
    """
    Hamming code. For a given redundancy :math:`m`, it is the linear block code (:class:`BlockCode`) with parity-check matrix whose columns are all the :math:`2^m - 1` nonzero binary :math:`m`-tuples. The Hamming code has the following parameters:

    - Length: :math:`n = 2^m - 1`
    - Redundancy: :math:`m`
    - Dimension: :math:`k = 2^m - m - 1`
    - Minimum distance: :math:`d = 3`

    This class constructs the code in systematic form, with the information set on the left.

    References: :cite:`Lin.Costello.04` (Sec 4.1)

    **Decoding methods**

    [[0]]

    **Notes**

    - For :math:`m = 2` it reduces to the repetition code (:class:`RepetitionCode`) of length :math:`3`.
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
    """
    def __init__(self, m):
        """
        Constructor for the class. It expects the following parameter:

        :code:`m` : :obj:`int`
            The redundancy :math:`m` of the code. Must satisfy :math:`m \\geq 2`.
        """
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
    Simplex (maximum-length) code. For a given dimension :math:`k`, it is the linear block code (:class:`BlockCode`) with generator matrix whose columns are all the :math:`2^k - 1` nonzero binary :math:`k`-tuples. The simplex code (also known as maximum-length code) has the following parameters:

    - Length: :math:`n = 2^k - 1`
    - Dimension: :math:`k`
    - Redundancy: :math:`m = 2^k - k - 1`
    - Minimum distance: :math:`d = 2^{k - 1}`

    This class constructs the code in systematic form, with the information set on the left.

    **Decoding methods**

    [[0]]

    **Notes**

    - For :math:`k = 2` it reduces to the single parity check code (:class:`SingleParityCheckCode`) of length :math:`3`.
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
    """
    def __init__(self, k):
        """
        Constructor for the class. It expects the following parameter:

        :code:`k` : :obj:`int`
            The dimension :math:`k` of the code. Must satisfy :math:`k \\geq 2`.
        """
        super().__init__(parity_submatrix=HammingCode._hamming_parity_submatrix(k).T)
        self._minimum_distance = 2**(k - 1)

    def __repr__(self):
        args = 'dimension={}'.format(self._dimension)
        return '{}({})'.format(self.__class__.__name__, args)


class GolayCode(BlockCode):
    """
    Binary Golay code. It has the following parameters:

    - Length: :math:`23`
    - Dimension: :math:`12`
    - Minimum distance: :math:`7`

    This class constructs the code in systematic form, with the information set on the left.

    **Decoding methods**

    [[0]]

    **Notes**

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
        """
        Constructor for the class. It expects no parameters.
        """
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
    Repetition code. For a given length :math:`n`, it is the linear block code (:class:`BlockCode`) whose only two codewords are :math:`00 \\cdots 0` and :math:`11 \\cdots 1`. The repetition code has the following parameters:

    - Length: :math:`n`
    - Dimension: :math:`k = 1`
    - Minimum distance: :math:`d = n`

    **Decoding methods**

    [[0]]

    **Notes**

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
        """
        Constructor for the class. It expects the following parameter:

        :code:`n` : :obj:`int`
            The length :math:`n` of the code. Must be a positive integer.
        """
        super().__init__(parity_submatrix=np.ones((1, n - 1), dtype=np.int))
        self._minimum_distance = n

    def __repr__(self):
        args = 'length={}'.format(self._length)
        return '{}({})'.format(self.__class__.__name__, args)

    @tag(name='Majority-logic', input_type='hard', target='codeword')
    def _decode_majority_logic(self, recvword):
        """
        Majority-logic decoder. A hard-decision decoder for Repetition codes only.
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
    Single parity check code. For a given length :math:`n`, it is the linear block code (:class:`BlockCode`) whose codewords are obtained by extending :math:`n - 1` information bits with a single parity-check bit. The repetition code has the following parameters:

    - Length: :math:`n`.
    - Dimension: :math:`k = n - 1`.
    - Minimum distance: :math:`d = 2`.

    **Decoding methods**

    [[0]]

    **Notes**

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
        """
        Constructor for the class. It expects the following parameter:

        :code:`n` : :obj:`int`
            The length :math:`n` of the code. Must be a positive integer.
        """
        super().__init__(parity_submatrix=np.ones((1, n - 1), dtype=np.int).T)
        self._minimum_distance = 2

    def __repr__(self):
        args = 'length={}'.format(self._length)
        return '{}({})'.format(self.__class__.__name__, args)

    @tag(name='Wagner', input_type='soft', target='codeword')
    def _decode_wagner(self, recvword):
        """
        Wagner decoder. A soft-decision decoder for SingleParityCheck codes only. See Costello, Forney: Channel Coding: The Road to Channel Capacity.
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
    Reed--Muller code. It is a linear block code (:obj:`BlockCode`) defined by two integers :math:`\\rho` and :math:`\\mu`, which must satisfy :math:`0 \\leq \\rho < \\mu`. See references for more details. The resulting code is denoted by :math:`\\mathrm{RM}(\\rho, \\mu)`, and has the following parameters:

    - Length: :math:`n = 2^{\\mu}`
    - Dimension: :math:`k = 1 + {\\mu \\choose 1} + \\cdots + {\\mu \\choose \\rho}`
    - Redundancy: :math:`m = 1 + {\\mu \\choose 1} + \\cdots + {\\mu \\choose \\mu - \\rho - 1}`
    - Minimum distance: :math:`d = 2^{\\mu - \\rho}`

    References: :cite:`Lin.Costello.04` (p. 105--114)

    **Decoding methods**

    [[0]]

    **Notes**

    - For :math:`\\rho = 0` it reduces to a repetition code (:class:`RepetitionCode`).
    - For :math:`\\rho = 1` it reduces to an extended simplex code (:class:`SimplexCode`).
    - For :math:`\\rho = \\mu - 2` it reduces to an extended Hamming code (:class:`HammingCode`).
    - For :math:`\\rho = \\mu - 1` it reduces to a single parity check code (:class:`SingleParityCheckCode`).

    .. rubric:: Examples

    >>> code = komm.ReedMullerCode(1, 5)
    >>> (code.length, code.dimension, code.minimum_distance)
    (32, 6, 16)
    """
    def __init__(self, rho, mu):
        """
        Constructor for the class. It expects the following parameters:

        :code:`rho` : :obj:`int`
            The parameter :math:`\\rho` of the code.

        :code:`mu` : :obj:`int`
            The parameter :math:`\\mu` of the code.

        The parameters must satisfy :math:`0 \\leq \\rho < \\mu`.
        """
        assert 0 <= rho < mu

        super().__init__(generator_matrix=ReedMullerCode._reed_muller_generator_matrix(rho, mu))
        self._minimum_distance = 2**(mu - rho)
        self._rho = rho
        self._mu = mu

    def __repr__(self):
        args = '{}, {}'.format(self._rho, self._mu)
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def rho(self):
        """
        The parameter :math:`\\rho` of the code. This property is read-only.
        """
        return self._rho

    @property
    def mu(self):
        """
        The parameter :math:`\\mu` of the code. This property is read-only.
        """
        return self._mu

    @staticmethod
    def _reed_muller_generator_matrix(rho, mu):
        """
        [1] Lin, Costello, 2Ed, p. 105--114. Assumes 0 <= rho < mu.
        """
        v = np.empty((mu, 2**mu), dtype=np.int)
        for i in range(mu):
            block = np.hstack((np.zeros(2**(mu - i - 1), dtype=np.int), np.ones(2**(mu - i - 1), dtype=np.int)))
            v[mu - i - 1] = np.tile(block, 2**i)

        G_list = []
        for ell in range(rho, 0, -1):
            for I in itertools.combinations(range(mu), ell):
                row = functools.reduce(np.multiply, v[I, :])
                G_list.append(row)
        G_list.append(np.ones(2**mu, dtype=np.int))

        return np.array(G_list, dtype=np.int)

    @functools.lru_cache(maxsize=None)
    def _reed_partitions(self):
        """
        Get Reed partitions from Reed-Muller generator matrix. See Lin, Costello, 2Ed, p. 105--114.
        """
        reed_partitions = []
        for ell in range(self._rho, -1, -1):
            binary_vectors_I = np.array(list(binary_iterator(ell)), dtype=np.int)
            binary_vectors_J = np.array(list(binary_iterator(self._mu - ell)), dtype=np.int)
            for I in itertools.combinations(range(self._mu), ell):
                E = np.setdiff1d(np.arange(self._mu), I, assume_unique=True)
                S = np.dot(binary_vectors_I, 2**np.array(I, dtype=np.int))
                Q = np.dot(binary_vectors_J, 2**np.array(E, dtype=np.int))
                reed_partitions.append(S[np.newaxis] + Q[np.newaxis].T)
        return reed_partitions

    @tag(name='Reed', input_type='hard', target='message')
    def _decode_reed(self, recvword):
        """
        Reed decoding algorithm for Reed--Muller codes. It's a majority-logic decoding algorithm. See Lin, Costello, 2Ed, p. 105--114, 439--440.
        """
        reed_partitions = self._reed_partitions()
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
        Weighted Reed decoding algorithm for Reed--Muller codes. See Lin, Costello, 2Ed, p. 440-442.
        """
        reed_partitions = self._reed_partitions()
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
    General binary cyclic code. A cyclic code is a linear block code (:class:`BlockCode`) such that, if :math:`c` is a codeword, then every cyclic shift of :math:`c` is also a codeword. It is characterized by its *generator polynomial* :math:`g(X)`, of degree :math:`m` (the redundancy of the code), and by its *parity-check polynomial* :math:`h(X)`, of degree :math:`k` (the dimension of the code). Those polynomials are related by :math:`g(X) h(X) = X^n + 1`, where :math:`n = k + m` is the length of the code. See references for more details.

    Examples of generator polynomials can be found in the table below.

    =======================  ==============================================  ======================================
    Code :math:`(n, k, d)`   Generator polynomial :math:`g(X)`               Integer representation
    =======================  ==============================================  ======================================
    Hamming :math:`(7,4,3)`  :math:`X^3 + X + 1`                             :code:`0b1011 = 0o13 = 11`
    Simplex :math:`(7,3,4)`  :math:`X^4 + X^2 + X +   1`                     :code:`0b10111 = 0o27 = 23`
    BCH :math:`(15,5,7)`     :math:`X^{10} + X^8 + X^5 + X^4 + X^2 + X + 1`  :code:`0b10100110111 = 0o2467 = 1335`
    Golay :math:`(23,12,7)`  :math:`X^{11} + X^9 + X^7 + X^6 + X^5 + X + 1`  :code:`0b101011100011 = 0o5343 = 2787`
    =======================  ==============================================  ======================================

    References: :cite:`Lin.Costello.04` (Chapter 5)

    **Decoding methods**

    [[0]]
    """
    def __init__(self, length, systematic=True, **kwargs):
        """
        Constructor for the class. It expects one of the following formats:

        **Via generator polynomial**

        `komm.CyclicCode(length, generator_polynomial=generator_polynomial, systematic=True)`

        :code:`generator_polynomial` : :obj:`BinaryPolynomial` or :obj:`int`
            The generator polynomial :math:`g(X)` of the code, of degree :math:`m` (the redundancy of the code), specified either as a :obj:`BinaryPolynomial` or as an :obj:`int` to be converted to the former.

        **Via parity-check polynomial**

        `komm.CyclicCode(length, parity_check_polynomial=parity_check_polynomial, systematic=True)`

        :code:`parity_check_polynomial` : :obj:`BinaryPolynomial` or :obj:`int`
            The parity-check polynomial :math:`h(X)` of the code, of degree :math:`k` (the dimension of the code), specified either as a :obj:`BinaryPolynomial` or as an :obj:`int` to be converted to the former.

        **Common parameters**

        :code:`length` : :obj:`int`
            The length :math:`n` of the code.

        :code:`systematic` : :obj:`bool`, optional
            Whether the encoder is systematic. Default is :code:`True`.

        .. rubric:: Examples

        >>> code = komm.CyclicCode(length=23, generator_polynomial=0b101011100011)  # Golay (23, 12)
        >>> (code.length, code.dimension, code.minimum_distance)
        (23, 12, 7)

        >>> code = komm.CyclicCode(length=23, parity_check_polynomial=0b1010010011111)  # Golay (23, 12)
        >>> (code.length, code.dimension, code.minimum_distance)
        (23, 12, 7)
        """
        self._length = length
        self._modulus = BinaryPolynomial.from_exponents([0, self._length])
        kwargs_set = set(kwargs.keys())
        if kwargs_set == {'generator_polynomial'}:
            self._generator_polynomial = BinaryPolynomial(kwargs['generator_polynomial'])
            self._parity_check_polynomial, remainder = divmod(self._modulus, self._generator_polynomial)
            if remainder != 0b0:
                raise ValueError("The generator polynomial must be a factor of X^n + 1")
            self._constructed_from = 'generator_polynomial'
        elif kwargs_set == {'parity_check_polynomial'}:
            self._parity_check_polynomial = BinaryPolynomial(kwargs['parity_check_polynomial'])
            self._generator_polynomial, remainder = divmod(self._modulus, self._parity_check_polynomial)
            if remainder != 0b0:
                raise ValueError("The parity-check polynomial must be a factor of X^n + 1")
            self._constructed_from = 'parity_check_polynomial'
        else:
            raise ValueError("Either specify 'generator_polynomial' or 'parity_check_polynomial'")
        self._dimension = self._parity_check_polynomial.degree
        self._redundancy = self._generator_polynomial.degree
        self._is_systematic = bool(systematic)
        if self._is_systematic:
            self._information_set = np.arange(self._redundancy, self._length)

    def __repr__(self):
        if self._constructed_from == 'generator_polynomial':
            args = 'length={}, generator_polynomial={}, systematic={}'.format(self._length, self._generator_polynomial, self._is_systematic)
        elif self._constructed_from == 'parity_check_polynomial':
            args = 'length={}, parity_check_polynomial={}, systematic={}'.format(self._length, self._parity_check_polynomial, self._is_systematic)
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def generator_polynomial(self):
        """
        The generator polynomial :math:`g(X)` of the cyclic code. It is a binary polynomial (:obj:`BinaryPolynomial`) of degree :math:`m`, where :math:`m` is the redundancy of the code.
        """
        return self._generator_polynomial

    @property
    def parity_check_polynomial(self):
        """
        The parity-check polynomial :math:`h(X)` of the cyclic code. It is a binary polynomial (:obj:`BinaryPolynomial`) of degree :math:`k`, where :math:`k` is the dimension of the code.
        """
        return self._parity_check_polynomial

    def meggitt_table(self):
        """
        Returns the Meggit table for the cyclic code. See :cite:`Xambo-Descamps.03` (Sec. 3.4) for more details.

        **Output:**

        :code:`meggitt_table` : :obj:`dict`
            A dictionary where the keys are syndromes and the values are error patterns.

        This is a cached method.

        .. rubric:: Examples

        >>> code = komm.CyclicCode(length=7, generator_polynomial=0b10111)
        >>> code.meggitt_table()
        {0b1011: 0b1000000,
         0b1010: 0b1000001,
         0b1001: 0b1000010,
         0b1111: 0b1000100,
         0b11: 0b1001000,
         0b1100: 0b1010000,
         0b101: 0b1100000}
        """
        if not hasattr(self, '_meggitt_table'):
            self._meggitt_table = {}
            for w in range(self.packing_radius):
                for idx in itertools.combinations(range(self._length - 1), w):
                    errorword_polynomial = BinaryPolynomial.from_exponents(list(idx) + [self._length - 1])
                    syndrome_polynomial = errorword_polynomial % self._generator_polynomial
                    self._meggitt_table[syndrome_polynomial] = errorword_polynomial
        return self._meggitt_table

    def _encode_cyclic_direct(self, message):
        """
        Encoder for cyclic codes. Direct, non-systematic method.
        """
        message_polynomial = BinaryPolynomial.from_coefficients(message)
        return (message_polynomial * self._generator_polynomial).coefficients(width=self._length)

    def _encode_cyclic_systematic(self, message):
        """
        Encoder for cyclic codes. Systematic method.
        """
        message_polynomial = BinaryPolynomial.from_coefficients(message)
        message_polynomial_shifted = message_polynomial << self._generator_polynomial.degree
        parity = message_polynomial_shifted % self._generator_polynomial
        return (message_polynomial_shifted + parity).coefficients(width=self._length)

    def _default_encoder(self):
        if self._is_systematic:
            return 'cyclic_systematic'
        else:
            return 'cyclic_direct'

    @property
    @functools.lru_cache(maxsize=None)
    def generator_matrix(self):
        n, k = self.length, self.dimension
        generator_matrix = np.empty((k, n), dtype=np.int)
        row = self._generator_polynomial.coefficients(width=n)
        for i in range(k):
            generator_matrix[i] = np.roll(row, i)
        return generator_matrix

    @property
    @functools.lru_cache(maxsize=None)
    def parity_check_matrix(self):
        n, k = self.length, self.dimension
        parity_check_matrix = np.empty((n - k, n), dtype=np.int)
        row = self._parity_check_polynomial.coefficients(width=n)[::-1]
        for i in range(n - k):
            parity_check_matrix[n - k - i - 1] = np.roll(row, -i)
        return parity_check_matrix

    @tag(name='Meggitt decoder', input_type='hard', target='codeword')
    def _decode_meggitt(self, recvword):
        """
        Meggitt decoder. See :cite:`Xambo-Descamps.03` (Sec. 3.4) for more details.
        """
        meggitt_table = self.meggitt_table()
        recvword_polynomial = BinaryPolynomial.from_coefficients(recvword)
        syndrome_polynomial = recvword_polynomial % self._generator_polynomial
        if syndrome_polynomial == 0:
            return recvword
        errorword_polynomial_hat = BinaryPolynomial(0)
        for j in range(self._length):
            if syndrome_polynomial in meggitt_table:
                errorword_polynomial_hat = meggitt_table[syndrome_polynomial] // (1 << j)
                break
            syndrome_polynomial = (syndrome_polynomial << 1) % self._generator_polynomial
        return (recvword_polynomial + errorword_polynomial_hat).coefficients(self._length)

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'meggitt'
        else:
            return super()._default_decoder(dtype)


class BCHCode(CyclicCode):
    """
    Bose--Chaudhuri--Hocquenghem (BCH) code. It is a cyclic code (:obj:`CyclicCode`) specified by two integers :math:`\\mu` and :math:`\\tau` which must satisfy :math:`1 \\leq \\tau < 2^{\mu - 1}`.  The parameter :math:`\\tau` is called the *designed error-correcting capability* of the BCH code; it will be internally replaced by the true error-correcting capability :math:`t` of the code. See references for more details. The resulting code is denoted by :math:`\\mathrm{BCH}(\\mu, \\tau)`, and has the following parameters:

    - Length: :math:`n = 2^{\\mu} - 1`
    - Dimension: :math:`k \geq n - \\mu \\tau`
    - Redundancy: :math:`m \\leq \\mu \\tau`
    - Minimum distance: :math:`d \\geq 2\\tau + 1`

    **Decoding methods**

    [[0]]

    References: :cite:`Lin.Costello.04` (Ch. 6)

    .. rubric:: Examples

    >>> code = komm.BCHCode(5, 3)
    >>> (code.length, code.dimension, code.minimum_distance)
    (31, 16, 7)
    >>> code.generator_polynomial
    0b1000111110101111

    >>> # TODO: Example of tau being replaced by its true value.
    """
    def __init__(self, mu, tau):
        """
        Constructor for the class. It expects the following parameters:

        :code:`mu` : :obj:`int`
            The parameter :math:`\\mu` of the code.

        :code:`tau` : :obj:`int`
            The designed error-correcting capability :math:`\\tau` of the BCH code. It will be internally replaced by the true error-correcting capability :math:`t` of the code.
        """
        assert 1 <= tau < 2**(mu - 1)

        field = BinaryFiniteExtensionField(mu)
        generator_polynomial, t = self._bch_code_generator_polynomial(field, tau)
        super().__init__(length=2**mu - 1, generator_polynomial=generator_polynomial)

        self._field = field
        self._mu = mu
        self._packing_radius = t
        self._minimum_distance = 2*t + 1

        alpha = field.primitive_element
        self._beta = [alpha**(i + 1) for i in range(2*t)]
        self._beta_minimal_polynomial = [b.minimal_polynomial() for b in self._beta]

    def __repr__(self):
        args = '{}, {}'.format(self._mu, self._packing_radius)
        return '{}({})'.format(self.__class__.__name__, args)

    @staticmethod
    def _bch_code_generator_polynomial(field, tau):
        """
        Assumes 1 <= tau < 2**(mu - 1). See :cite:`Lin.Costello.04` (p. 194--195)
        """
        alpha = field.primitive_element

        t = tau
        lcm_set = {(alpha**(2*i + 1)).minimal_polynomial() for i in range(t)}
        while True:
            if (alpha**(2*t + 1)).minimal_polynomial() not in lcm_set:
                break
            t += 1
        generator_polynomial = functools.reduce(operator.mul, lcm_set)

        return generator_polynomial, t

    def _bch_general_decoder(self, recvword, syndrome_computer, key_equation_solver, root_finder):
        """
        General BCH decoder. See :cite:`Lin.Costello.04` (p. 205--209).
        """
        recvword_polynomial = BinaryPolynomial.from_coefficients(recvword)
        syndrome_polynomial = syndrome_computer(recvword_polynomial)
        if np.all([x == self._field(0) for x in syndrome_polynomial]):
            return recvword
        error_location_polynomial = key_equation_solver(syndrome_polynomial)
        error_locations = [e.inverse().logarithm() for e in root_finder(error_location_polynomial)]
        errorword = np.bincount(error_locations, minlength=recvword.size)
        return np.bitwise_xor(recvword, errorword)

    def _bch_syndrome(self, recvword_polynomial):
        """
        BCH syndrome computation. See :cite:`Lin.Costello.04` (p. 205--209).
        """
        syndrome_polynomial = np.empty(len(self._beta), dtype=np.object)
        for i, (b, b_min_polynomial) in enumerate(zip(self._beta, self._beta_minimal_polynomial)):
            syndrome_polynomial[i] = (recvword_polynomial % b_min_polynomial).evaluate(b)
        return syndrome_polynomial

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

    def _berlekamp_algorithm(self, syndrome_polynomial):
        """
        Berlekamp's iterative procedure for finding the error-location polynomial of a BCH code. See  :cite:`Lin.Costello.04` (p. 209--212) and :cite:`Ryan.Lin.09` (p. 114-121).
        """
        field = self._field
        t = self._packing_radius

        sigma = {-1: np.array([field(1)], dtype=np.object), 0: np.array([field(1)], dtype=np.object)}
        discrepancy = {-1: field(1), 0: syndrome_polynomial[0]}
        degree = {-1: 0, 0: 0}

        #TODO: This mu is not the same as the mu in __init__...
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
                discrepancy[mu + 1] = syndrome_polynomial[mu + 1]
                for idx in range(1, degree[mu + 1] + 1):
                    discrepancy[mu + 1] += sigma[mu + 1][idx] * syndrome_polynomial[mu + 1 - idx]

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
