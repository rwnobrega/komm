import functools
import itertools

import numpy as np

from .._algebra import BinaryPolynomial
from .._aux import tag
from .BlockCode import BlockCode


class CyclicCode(BlockCode):
    r"""
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

    References:

        1. :cite:`Lin.Costello.04` (Chapter 5)

    .. rubric:: Decoding methods

    [[decoding_methods]]
    """

    def __init__(self, length, systematic=True, **kwargs):
        r"""
        Constructor for the class. It expects one of the following formats:

        **Via generator polynomial**

        `komm.CyclicCode(length, generator_polynomial=generator_polynomial, systematic=True)`

        Parameters:

            generator_polynomial (:obj:`BinaryPolynomial` or :obj:`int`): The generator polynomial :math:`g(X)` of the code, of degree :math:`m` (the redundancy of the code), specified either as a :obj:`BinaryPolynomial` or as an :obj:`int` to be converted to the former.

        **Via parity-check polynomial**

        `komm.CyclicCode(length, parity_check_polynomial=parity_check_polynomial, systematic=True)`

            parity_check_polynomial (:obj:`BinaryPolynomial` or :obj:`int`): The parity-check polynomial :math:`h(X)` of the code, of degree :math:`k` (the dimension of the code), specified either as a :obj:`BinaryPolynomial` or as an :obj:`int` to be converted to the former.

        The following parameters are common to both formats:

        Parameters:

            length (:obj:`int`): The length :math:`n` of the code.

            systematic (:obj:`bool`, optional): Whether the encoder is systematic. Default is :code:`True`.

        Examples:

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
        if kwargs_set == {"generator_polynomial"}:
            self._generator_polynomial = BinaryPolynomial(kwargs["generator_polynomial"])
            self._parity_check_polynomial, remainder = divmod(self._modulus, self._generator_polynomial)
            if remainder != 0b0:
                raise ValueError("The generator polynomial must be a factor of X^n + 1")
            self._constructed_from = "generator_polynomial"
        elif kwargs_set == {"parity_check_polynomial"}:
            self._parity_check_polynomial = BinaryPolynomial(kwargs["parity_check_polynomial"])
            self._generator_polynomial, remainder = divmod(self._modulus, self._parity_check_polynomial)
            if remainder != 0b0:
                raise ValueError("The parity-check polynomial must be a factor of X^n + 1")
            self._constructed_from = "parity_check_polynomial"
        else:
            raise ValueError("Either specify 'generator_polynomial' or 'parity_check_polynomial'")
        self._dimension = self._parity_check_polynomial.degree
        self._redundancy = self._generator_polynomial.degree
        self._is_systematic = bool(systematic)
        if self._is_systematic:
            self._information_set = np.arange(self._redundancy, self._length)

    def __repr__(self):
        if self._constructed_from == "generator_polynomial":
            args = "length={}, generator_polynomial={}, systematic={}".format(
                self._length, self._generator_polynomial, self._is_systematic
            )
        else:  # if self._constructed_from == "parity_check_polynomial":
            args = "length={}, parity_check_polynomial={}, systematic={}".format(
                self._length, self._parity_check_polynomial, self._is_systematic
            )
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def generator_polynomial(self):
        r"""
        The generator polynomial :math:`g(X)` of the cyclic code. It is a binary polynomial (:obj:`BinaryPolynomial`) of degree :math:`m`, where :math:`m` is the redundancy of the code.
        """
        return self._generator_polynomial

    @property
    def parity_check_polynomial(self):
        r"""
        The parity-check polynomial :math:`h(X)` of the cyclic code. It is a binary polynomial (:obj:`BinaryPolynomial`) of degree :math:`k`, where :math:`k` is the dimension of the code.
        """
        return self._parity_check_polynomial

    @functools.cached_property
    def meggitt_table(self):
        r"""
        The Meggitt table for the cyclic code. It is a dictionary where the keys are syndromes and the values are error patterns. See :cite:`Xambo-Descamps.03` (Sec. 3.4) for more details.
        """
        meggitt_table = {}
        for w in range(self.packing_radius + 1):
            for idx in itertools.combinations(range(self._length - 1), w):
                errorword_polynomial = BinaryPolynomial.from_exponents(list(idx) + [self._length - 1])
                syndrome_polynomial = errorword_polynomial % self._generator_polynomial
                meggitt_table[syndrome_polynomial] = errorword_polynomial
        return meggitt_table

    def _encode_cyclic_direct(self, message):
        r"""
        Encoder for cyclic codes. Direct, non-systematic method.
        """
        message_polynomial = BinaryPolynomial.from_coefficients(message)
        return (message_polynomial * self._generator_polynomial).coefficients(width=self._length)

    def _encode_cyclic_systematic(self, message):
        r"""
        Encoder for cyclic codes. Systematic method.
        """
        message_polynomial = BinaryPolynomial.from_coefficients(message)
        message_polynomial_shifted = message_polynomial << self._generator_polynomial.degree
        parity = message_polynomial_shifted % self._generator_polynomial
        return (message_polynomial_shifted + parity).coefficients(width=self._length)

    def _default_encoder(self):
        if self._is_systematic:
            return "cyclic_systematic"
        else:
            return "cyclic_direct"

    @functools.cached_property
    def generator_matrix(self):
        n, k = self.length, self.dimension
        generator_matrix = np.empty((k, n), dtype=int)
        row = self._generator_polynomial.coefficients(width=n)
        for i in range(k):
            generator_matrix[i] = np.roll(row, i)
        return generator_matrix

    @functools.cached_property
    def parity_check_matrix(self):
        n, k = self.length, self.dimension
        parity_check_matrix = np.empty((n - k, n), dtype=int)
        row = self._parity_check_polynomial.coefficients(width=n)[::-1]
        for i in range(n - k):
            parity_check_matrix[n - k - i - 1] = np.roll(row, -i)
        return parity_check_matrix

    @tag(name="Meggitt decoder", input_type="hard", target="codeword")
    def _decode_meggitt(self, recvword):
        r"""
        Meggitt decoder. See :cite:`Xambo-Descamps.03` (Sec. 3.4) for more details.
        """
        meggitt_table = self.meggitt_table
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
        if dtype == int:
            return "meggitt"
        else:
            return super()._default_decoder(dtype)
