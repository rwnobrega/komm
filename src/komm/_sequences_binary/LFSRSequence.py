from typing_extensions import Self

from .._algebra.BinaryPolynomial import BinaryPolynomial, default_primitive_polynomial
from .BinarySequence import BinarySequence
from .sequences import lfsr_sequence


class LFSRSequence(BinarySequence):
    r"""
    Linear-feedback shift register (LFSR) sequence. It is a [binary sequence](/ref/BinarySequence) obtained from the output of a LFSR. The LFSR feedback taps are specified as a [binary polynomial](/ref/BinaryPolynomial) $p(X)$ of degree $n$, called the *feedback polynomial*. More specifically: if bit $i$ of the LFSR is tapped, for $i \in [1 : n]$, then the coefficient of $X^i$ in $p(X)$ is $1$; otherwise, it is $0$; moreover, the coefficient of $X^0$ in $p(X)$ is always $1$. For example, the feedback polynomial corresponding to the LFSR in the figure below is $p(X) = X^5 + X^2 + 1$, whose integer representation is `0b100101`.

    <figure markdown>
      ![Linear-feedback shift register example.](/figures/lfsr_5_2.svg)
    </figure>

    The start state of the machine is specified by the so called *start state polynomial*. More specifically, the coefficient of $X^i$ in the start state polynomial is equal to the initial value of bit $i$ of the LFSR. For more details, see [Wikipedia: Linear-feedback shift register](https://en.wikipedia.org/wiki/Linear-feedback_shift_register) and [Wikipedia: Maximum-length sequence](https://en.wikipedia.org/wiki/Maximum_length_sequence).

    The default constructor of this takes the following parameters:

    Parameters:
        feedback_polynomial: The feedback polynomial $p(X)$ of the LFSR, specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former.

        start_state_polynomial: The start state polynomial of the LFSR, specified either as a [binary polynomial](/ref/BinaryPolynomial) or as an integer to be converted to the former. The default value is `0b1`.

    Examples:
        >>> lfsr = komm.LFSRSequence(feedback_polynomial=0b100101)
        >>> lfsr.bit_sequence
        array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
        >>> lfsr.cyclic_autocorrelation()
        array([31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    <h2>Maximum-length sequences</h2>

    If the feedback polynomial $p(X)$ is primitive, then the corresponding LFSR sequence will be a *maximum-length sequence*. Such sequences have the following cyclic autocorrelation:
    $$
        \tilde{R}[\ell] =
        \begin{cases}
            L, & \ell = 0, \, \pm L, \, \pm 2L, \ldots, \\\\
            -1, & \text{otherwise},
        \end{cases}
    $$
    where $L$ is the length of the sequence. See the class method [`maximum_length_sequence`](./#maximum_length_sequence) for a convenient way to construct a maximum-length sequence.
    """

    def __init__(
        self,
        feedback_polynomial: BinaryPolynomial | int,
        start_state_polynomial: BinaryPolynomial | int = 0b1,
    ) -> None:
        self.feedback_polynomial = BinaryPolynomial(feedback_polynomial)
        self.start_state_polynomial = BinaryPolynomial(start_state_polynomial)
        super().__init__(
            bit_sequence=lfsr_sequence(
                self.feedback_polynomial, self.start_state_polynomial
            )
        )

    @classmethod
    def maximum_length_sequence(
        cls, degree: int, start_state_polynomial: BinaryPolynomial | int = 0b1
    ) -> Self:
        r"""
        Constructs a maximum-length sequences of a given degree. The feedback polynomial $p(X)$ is chosen from [the list of default primitive polynomials](/resources/primitive-polynomials).

        Parameters:
            degree: The degree $n$ of the maximum-length-sequence. Only degrees in the range $[1 : 24]$ are implemented.

            start_state_polynomial: See the corresponding parameter of the default constructor.

        Examples:
            >>> komm.LFSRSequence.maximum_length_sequence(degree=5)
            LFSRSequence(feedback_polynomial=0b100101, start_state_polynomial=0b1)
        """
        return cls(
            feedback_polynomial=default_primitive_polynomial(degree),
            start_state_polynomial=start_state_polynomial,
        )

    def __repr__(self) -> str:
        args = (
            f"feedback_polynomial={self.feedback_polynomial}, "
            f"start_state_polynomial={self.start_state_polynomial}"
        )
        return f"{self.__class__.__name__}({args})"
