import numpy as np

from .._algebra import BinaryPolynomial
from .BinarySequence import BinarySequence


class LFSRSequence(BinarySequence):
    r"""
    Linear-feedback shift register (LFSR) sequence. It is a [binary sequence](/ref/BinarySequence) obtained from the output of a LFSR. The LFSR feedback taps are specified as a binary polynomial $p(X)$ of degree $n$, called the *feedback polynomial*. More specifically: if bit $i$ of the LFSR is tapped, for $i \in [1 : n]$, then the coefficient of $X^i$ in $p(X)$ is $1$; otherwise, it is $0$; moreover, the coefficient of $X^0$ in $p(X)$ is always $1$. For example, the feedback polynomial corresponding to the LFSR in the figure below is $p(X) = X^5 + X^2 + 1$, whose integer representation is `0b100101`.

    <figure markdown>
      ![Linear-feedback shift register example.](/figures/lfsr_5_2.svg)
    </figure>

    The start state of the machine is specified by the so called *start state polynomial*. More specifically, the coefficient of $X^i$ in the start state polynomial is equal to the initial value of bit $i$ of the LFSR.

    <h2>Maximum-length sequences</h2>

    If the feedback polynomial $p(X)$ is primitive, then the corresponding LFSR sequence will be a *maximum-length sequence* (MLS). Such sequences have the following cyclic autocorrelation:
    $$
        \tilde{R}[\ell] =
        \begin{cases}
            L, & \ell = 0, \, \pm L, \, \pm 2L, \ldots, \\\\
            -1, & \text{otherwise},
        \end{cases}
    $$
    where $L$ is the length of the sequence.

    References:

        1. https://en.wikipedia.org/wiki/Linear-feedback_shift_register
        2. https://en.wikipedia.org/wiki/Maximum_length_sequence
    """

    def __init__(self, feedback_polynomial, start_state_polynomial=0b1):
        r"""
        Default constructor for the class.

        Parameters:

            feedback_polynomial (BinaryPolynomial | int): The feedback polynomial of the LFSR, specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former.

            start_state_polynomial (Optional[BinaryPolynomial | int]): The start state polynomial of the LFSR, specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former. The default value is `0b1`.

        Examples:

            >>> lfsr = komm.LFSRSequence(feedback_polynomial=0b100101)
            >>> lfsr.bit_sequence  #doctest: +NORMALIZE_WHITESPACE
            array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
            >>> lfsr.cyclic_autocorrelation()  #doctest: +NORMALIZE_WHITESPACE
            array([31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

        See also the class method [`maximum_length_sequence`](./#maximum_length_sequence) for a more convenient way to construct a maximum-length sequence.
        """
        self._feedback_polynomial = BinaryPolynomial(feedback_polynomial)
        self._start_state_polynomial = BinaryPolynomial(start_state_polynomial)
        super().__init__(
            bit_sequence=self._lfsr_sequence(
                self._feedback_polynomial, self._start_state_polynomial
            )
        )

    @classmethod
    def maximum_length_sequence(cls, degree, start_state_polynomial=0b1):
        r"""
        Constructs a maximum-length sequences (MLS) of a given degree. The feedback polynomial $p(X)$ is chosen according to the following table of primitive polynomials.

        | Degree $n$ | Feedback polynomial $p(X)$ | Degree $n$ | Feedback polynomial $p(X)$ |
        | :--------: | -------------------------- | :--------: | -------------------------- |
        | $1$        | `0b11`                     | $9$        | `0b1000010001`             |
        | $2$        | `0b111`                    | $10$       | `0b10000001001`            |
        | $3$        | `0b1011`                   | $11$       | `0b100000000101`           |
        | $4$        | `0b10011`                  | $12$       | `0b1000001010011`          |
        | $5$        | `0b100101`                 | $13$       | `0b10000000011011`         |
        | $6$        | `0b1000011`                | $14$       | `0b100010001000011`        |
        | $7$        | `0b10001001`               | $15$       | `0b1000000000000011`       |
        | $8$        | `0b100011101`              | $16$       | `0b10001000000001011`      |

        Parameters:

            degree (int): The degree $n$ of the MLS. Only degrees in the range $[1 : 16]$ are implemented.

            start_state_polynomial (Optional[BinaryPolynomial | int]): See the corresponding parameter of the default constructor.

        Examples:

            >>> komm.LFSRSequence.maximum_length_sequence(degree=5)
            LFSRSequence(feedback_polynomial=0b100101)
        """
        return cls(
            feedback_polynomial=cls._default_primitive_polynomial(degree),
            start_state_polynomial=start_state_polynomial,
        )

    def __repr__(self):
        args = "feedback_polynomial={}".format(self._feedback_polynomial)
        if self._start_state_polynomial != BinaryPolynomial(0b1):
            args += ", start_state_polynomial={}".format(self._start_state_polynomial)
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def feedback_polynomial(self):
        r"""
        The feedback polynomial $p(X)$ of the LFSR.
        """
        return self._feedback_polynomial

    @property
    def start_state_polynomial(self):
        r"""
        The start state polynomial of the LFSR.
        """
        return self._start_state_polynomial

    @staticmethod
    def _lfsr_sequence(feedback_polynomial, start_state_polynomial):
        taps = (feedback_polynomial + BinaryPolynomial(1)).exponents()
        start_state = start_state_polynomial.coefficients(
            width=feedback_polynomial.degree
        )
        m = taps[-1]
        L = 2**m - 1
        state = np.copy(start_state)
        code = np.empty(L, dtype=int)
        for i in range(L):
            code[i] = state[-1]
            state[-1] = np.count_nonzero(state[taps - 1]) % 2
            state = np.roll(state, 1)
        return code

    @staticmethod
    def _default_primitive_polynomial(degree):
        if degree < 1 or degree > 16:
            raise ValueError("Only degrees in the range [1 : 16] are implemented.")
        return {
            1: 0b11,
            2: 0b111,
            3: 0b1011,
            4: 0b10011,
            5: 0b100101,
            6: 0b1000011,
            7: 0b10001001,
            8: 0b100011101,
            9: 0b1000010001,
            10: 0b10000001001,
            11: 0b100000000101,
            12: 0b1000001010011,
            13: 0b10000000011011,
            14: 0b100010001000011,
            15: 0b1000000000000011,
            16: 0b10001000000001011,
        }[degree]
