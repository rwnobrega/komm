import functools
import itertools

import numpy as np

from .._algebra.util import null_matrix, right_inverse
from .._aux import tag
from .._util import binlist2int, int2binlist


class BlockCode:
    r"""
    General binary linear block code. It is characterized by its *generator matrix* $G$, a binary $k \times n$ matrix, and by its *parity-check matrix* $H$, a binary $m \times n$ matrix. Those matrix are related by $G H^\top = 0$. The parameters $k$, $m$, and $n$ are called the code *dimension*, *redundancy*, and *length*, respectively, and are related by $k + m = n$.

    References:

        1. :cite:`Lin.Costello.04` (Ch. 3).

    .. rubric:: Decoding methods

    [[decoding_methods]]

    Examples:

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
        >>> code.codeword_table
        array([[0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 1],
               [0, 1, 0, 1, 0, 1],
               [1, 1, 0, 1, 1, 0],
               [0, 0, 1, 1, 1, 0],
               [1, 0, 1, 1, 0, 1],
               [0, 1, 1, 0, 1, 1],
               [1, 1, 1, 0, 0, 0]])
        >>> code.codeword_weight_distribution
        array([1, 0, 0, 4, 3, 0, 0])
        >>> code.coset_leader_table
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1],
               [0, 1, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0],
               [1, 0, 0, 1, 0, 0]])
        >>> code.coset_leader_weight_distribution
        array([1, 6, 1, 0, 0, 0, 0])
        >>> (code.packing_radius, code.covering_radius)
        (1, 2)
    """

    def __init__(self, **kwargs):
        r"""
        Constructor for the class. It expects one of the following formats:

        **Via generator matrix**

        `komm.BlockCode(generator_matrix=generator_matrix)`

        Parameters:

            generator_matrix (2D-array of :obj:`int`): Generator matrix $G$ for the code, which is a $k \times n$ binary matrix.

        **Via parity-check matrix**

        `komm.BlockCode(parity_check_matrix=parity_check_matrix)`

        Parameters:

            parity_check_matrix (2D-array of :obj:`int`): Parity-check matrix $H$ for the code, which is an $m \times n$ binary matrix.

        **Via parity submatrix and information set**

        `komm.BlockCode(parity_submatrix=parity_submatrix, information_set=information_set)`

        Parameters:

            parity_submatrix (2D-array of :obj:`int`): Parity submatrix $P$ for the code, which is a $k \times m$ binary matrix.

            information_set ((1D-array of :obj:`int`) or :obj:`str`, optional): Either an array containing the indices of the information positions, which must be a $k$-sublist of $[0 : n)$, or one of the strings `'left'` or `'right'`. The default value is `'left'`.
        """
        if "generator_matrix" in kwargs:
            self._init_from_generator_matrix(**kwargs)
        elif "parity_check_matrix" in kwargs:
            self._init_from_parity_check_matrix(**kwargs)
        elif "parity_submatrix" in kwargs:
            self._init_from_parity_submatrix(**kwargs)
        else:
            raise ValueError("Either specify 'generator_matrix' or 'parity_check_matrix' or 'parity_submatrix'")

    def _init_from_generator_matrix(self, generator_matrix):
        self._generator_matrix = np.array(generator_matrix, dtype=int) % 2
        self._dimension, self._length = self._generator_matrix.shape
        self._redundancy = self._length - self._dimension
        self._is_systematic = False
        self._constructed_from = "generator_matrix"

    def _init_from_parity_check_matrix(self, parity_check_matrix):
        self._parity_check_matrix = np.array(parity_check_matrix, dtype=int) % 2
        self._redundancy, self._length = self._parity_check_matrix.shape
        self._dimension = self._length - self._redundancy
        self._is_systematic = False
        self._constructed_from = "parity_check_matrix"

    def _init_from_parity_submatrix(self, parity_submatrix, information_set="left"):
        self._parity_submatrix = np.array(parity_submatrix, dtype=int) % 2
        self._dimension, self._redundancy = self._parity_submatrix.shape
        self._length = self._dimension + self._redundancy
        if information_set == "left":
            information_set = np.arange(self._dimension)
        elif information_set == "right":
            information_set = np.arange(self._redundancy, self._length)
        self._information_set = np.array(information_set, dtype=int)
        if (
            self._information_set.size != self._dimension
            or self._information_set.min() < 0
            or self._information_set.max() > self._length
        ):
            raise ValueError("Parameter 'information_set' must be a 'k'-subset of 'range(n)'")
        self._parity_set = np.setdiff1d(np.arange(self._length), self._information_set)
        self._generator_matrix = np.empty((self._dimension, self._length), dtype=int)
        self._generator_matrix[:, self._information_set] = np.eye(self._dimension, dtype=int)
        self._generator_matrix[:, self._parity_set] = self._parity_submatrix
        self._parity_check_matrix = np.empty((self._redundancy, self._length), dtype=int)
        self._parity_check_matrix[:, self._information_set] = self._parity_submatrix.T
        self._parity_check_matrix[:, self._parity_set] = np.eye(self._redundancy, dtype=int)
        self._is_systematic = True
        self._constructed_from = "parity_submatrix"

    def __repr__(self):
        if self._constructed_from == "generator_matrix":
            args = "generator_matrix={}".format(self._generator_matrix.tolist())
        elif self._constructed_from == "parity_check_matrix":
            args = "parity_check_matrix={}".format(self._parity_check_matrix.tolist())
        elif self._constructed_from == "parity_submatrix":
            args = "parity_submatrix={}, information_set={}".format(
                self._parity_submatrix.tolist(), self._information_set.tolist()
            )
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def length(self):
        r"""
        The length $n$ of the code. This property is read-only.
        """
        return self._length

    @property
    def dimension(self):
        r"""
        The dimension $k$ of the code. This property is read-only.
        """
        return self._dimension

    @property
    def redundancy(self):
        r"""
        The redundancy $m$ of the code. This property is read-only.
        """
        return self._redundancy

    @property
    def rate(self):
        r"""
        The rate $R = k/n$ of the code. This property is read-only.
        """
        return self._dimension / self._length

    @functools.cached_property
    def minimum_distance(self):
        r"""
        The minimum distance $d$ of the code. This is equal to the minimum Hamming weight of the non-zero codewords. This property is read-only.
        """
        try:
            return self._minimum_distance
        except AttributeError:
            return np.flatnonzero(self.codeword_weight_distribution)[1]

    @functools.cached_property
    def packing_radius(self):
        r"""
        The packing radius of the code. This is also called the *error-correcting capability* of the code, and is equal to $\lfloor (d - 1) / 2 \rfloor$. This property is read-only.
        """
        return (self.minimum_distance - 1) // 2

    @functools.cached_property
    def covering_radius(self):
        r"""
        The covering radius of the code. This is equal to the maximum Hamming weight of the coset leaders. This property is read-only.
        """
        return np.flatnonzero(self.coset_leader_weight_distribution)[-1]

    @functools.cached_property
    def generator_matrix(self):
        r"""
        The generator matrix $G$ of the code. It as a $k \times n$ binary matrix, where $k$ is the code dimension, and $n$ is the code length. This property is read-only.
        """
        try:
            return self._generator_matrix
        except AttributeError:
            return null_matrix(self._parity_check_matrix)

    @functools.cached_property
    def parity_check_matrix(self):
        r"""
        The parity-check matrix $H$ of the code. It as an $m \times n$ binary matrix, where $m$ is the code redundancy, and $n$ is the code length. This property is read-only.
        """
        try:
            return self._parity_check_matrix
        except AttributeError:
            return null_matrix(self._generator_matrix)

    @functools.cached_property
    def codeword_table(self):
        r"""
        The codeword table of the code. This is a $2^k \times n$ matrix whose rows are all the codewords. The codeword in row $i$ corresponds to the message whose binary representation (MSB in the right) is $i$.
        """
        codeword_table = np.empty([2**self._dimension, self._length], dtype=int)
        for i in range(2**self._dimension):
            message = int2binlist(i, width=self._dimension)
            codeword_table[i] = self.encode(message)
        return codeword_table

    @functools.cached_property
    def codeword_weight_distribution(self):
        r"""
        The codeword weight distribution of the code. This is an array of shape $(n + 1)$ in which element in position $w$ is equal to the number of codewords of Hamming weight $w$, for $w \in [0 : n]$.
        """
        try:
            return self._codeword_weight_distribution
        except AttributeError:
            return np.bincount(np.sum(self.codeword_table, axis=1), minlength=self._length + 1)

    @functools.cached_property
    def coset_leader_table(self):
        r"""
        The coset leader table of the code. This is a $2^m \times n$ matrix whose rows are all the coset leaders. The coset leader in row $i$ corresponds to the syndrome whose binary representation (MSB in the right) is $i$. This may be used as a LUT for syndrome-based decoding.
        """
        coset_leader_table = np.empty([2**self._redundancy, self._length], dtype=int)
        taken = []
        for w in range(self._length + 1):
            for idx in itertools.combinations(range(self._length), w):
                errorword = np.zeros(self._length, dtype=int)
                errorword[list(idx)] = 1
                syndrome = np.dot(errorword, self.parity_check_matrix.T) % 2
                syndrome_int = binlist2int(syndrome)
                if syndrome_int not in taken:
                    coset_leader_table[syndrome_int] = np.array(errorword)
                    taken.append(syndrome_int)
                if len(taken) == 2**self.redundancy:
                    break
        return coset_leader_table

    @functools.cached_property
    def coset_leader_weight_distribution(self):
        r"""
        The coset leader weight distribution of the code. This is an array of shape $(n + 1)$ in which element in position $w$ is equal to the number of coset leaders of weight $w$, for $w \in [0 : n]$.
        """
        try:
            return self._coset_leader_weight_distribution
        except AttributeError:
            return np.bincount(np.sum(self.coset_leader_table, axis=1), minlength=self._length + 1)

    @functools.cached_property
    def _generator_matrix_right_inverse(self):
        return right_inverse(self.generator_matrix)

    def encode(self, message, method=None):
        r"""
        Encodes a given message to its corresponding codeword.

        Parameters:

            message (1D-array of :obj:`int`): The message to be encoded. Its length must be $k$.

            method (:obj:`str`, optional): The encoding method to be used.

        Returns:

            codeword (1D-array of :obj:`int`): The codeword corresponding to `message`. Its length is equal to $n$.
        """
        message = np.array(message)

        if message.size != self._dimension:
            raise ValueError("Length of 'message' must be equal to the code dimension")

        if method is None:
            method = self._default_encoder()

        encoder = getattr(self, "_encode_" + method)
        codeword = encoder(message)

        return codeword

    def _encode_generator_matrix(self, message):
        codeword = np.dot(message, self.generator_matrix) % 2
        return codeword

    def _encode_systematic_generator_matrix(self, message):
        codeword = np.empty(self._length, dtype=int)
        codeword[self._information_set] = message
        codeword[self._parity_set] = np.dot(message, self._parity_submatrix) % 2
        return codeword

    def _default_encoder(self):
        if self._is_systematic:
            return "systematic_generator_matrix"
        else:
            return "generator_matrix"

    def message_from_codeword(self, codeword):
        r"""
        Returns the message corresponding to a given codeword. In other words, applies the inverse encoding map.

        Parameters:

            codeword (1D-array of :obj:`int`): A codeword from the code. Its length must be $n$.

        Returns:

            message (1D-array of :obj:`int`): The message corresponding to `codeword`. Its length is equal to $k$.
        """
        if self._is_systematic:
            return codeword[self._information_set]
        else:
            return np.dot(codeword, self._generator_matrix_right_inverse) % 2

    def decode(self, recvword, method=None, **kwargs):
        r"""
        Decodes a received word to a message.

        Parameters:

            recvword (1D-array of (:obj:`int` or :obj:`float`)): The word to be decoded. If using a hard-decision decoding method, then the elements of the array must be bits (integers in $\\{ 0, 1 \\}$). If using a soft-decision decoding method, then the elements of the array must be soft-bits (floats standing for log-probability ratios, in which positive values represent bit $0$ and negative values represent bit $1$). Its length must be $n$.

            method (:obj:`str`, optional): The decoding method to be used.

            **kwargs: Keyword arguments to be passed to the decoding method.

        Returns:

            message_hat (1D-array of :obj:`int`): The message decoded from `recvword`. Its length is equal to $k$.
        """
        recvword = np.array(recvword)

        if recvword.size != self._length:
            raise ValueError("Length of 'recvword' must be equal to the code length")

        if method is None:
            method = self._default_decoder(recvword.dtype)

        decoder = getattr(self, "_decode_" + method)

        if decoder.target == "codeword":
            message_hat = self.message_from_codeword(decoder(recvword, **kwargs))
        elif decoder.target == "message":
            message_hat = decoder(recvword, **kwargs)

        return message_hat

    @tag(name="Exhaustive search (hard-decision)", input_type="hard", target="codeword")
    def _decode_exhaustive_search_hard(self, recvword):
        r"""
        Exhaustive search minimum distance hard decoder. Hamming distance.
        """
        codewords = self.codeword_table
        metrics = np.count_nonzero(recvword != codewords, axis=1)
        codeword_hat = codewords[np.argmin(metrics)]
        return codeword_hat

    @tag(name="Exhaustive search (soft-decision)", input_type="soft", target="codeword")
    def _decode_exhaustive_search_soft(self, recvword):
        r"""
        Exhaustive search minimum distance soft decoder. Euclidean distance.
        """
        codewords = self.codeword_table
        metrics = np.dot(recvword, codewords.T)
        codeword_hat = codewords[np.argmin(metrics)]
        return codeword_hat

    @tag(name="Syndrome table", input_type="hard", target="codeword")
    def _decode_syndrome_table(self, recvword):
        r"""
        Syndrome table decoder.
        """
        coset_leader_table = self.coset_leader_table
        syndrome = np.dot(recvword, self.parity_check_matrix.T) % 2
        syndrome_int = binlist2int(syndrome)
        errorword_hat = coset_leader_table[syndrome_int]
        codeword_hat = np.bitwise_xor(recvword, errorword_hat)
        return codeword_hat

    def _default_decoder(self, dtype):
        if dtype == int:
            if self._dimension >= self._redundancy:
                return "syndrome_table"
            else:
                return "exhaustive_search_hard"
        elif dtype == float:
            return "exhaustive_search_soft"

    @staticmethod
    def _extended_parity_submatrix(parity_submatrix):
        last_column = (1 + np.sum(parity_submatrix, axis=1)) % 2
        extended_parity_submatrix = np.hstack([parity_submatrix, last_column[np.newaxis].T])
        return extended_parity_submatrix

    @classmethod
    def _available_decoding_methods(cls):
        table = []
        for name in dir(cls):
            if name.startswith("_decode_"):
                identifier = name[8:]
                method = getattr(cls, name)
                table.append([method.name, "`{}`".format(identifier), method.input_type])
        return table

    @classmethod
    def _process_docstring(cls):
        table = cls._available_decoding_methods()
        indent = " " * 4
        rst = ".. csv-table::\n"
        rst += "{indent}   :header: Method, Identifier, Input type\n".format(indent=indent)
        rst += "{indent}   :widths: 5, 5, 2\n\n".format(indent=indent)
        for row in table:
            rst += "{indent}   {row}\n".format(indent=indent, row=", ".join(row))
        cls.__doc__ = cls.__doc__.replace("[[decoding_methods]]", rst)
