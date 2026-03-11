import operator
from functools import reduce

from typeguard import typechecked

from .._algebra.BinaryPolynomial import BinaryPolynomial
from .._algebra.FiniteBifield import FiniteBifield, FiniteBifieldElement
from .CyclicCode import CyclicCode


@typechecked
class BCHCode(CyclicCode):
    r"""
    Bose–Ray-Chaudhuri–Hocquenghem (BCH) code. For given parameters $\mu \geq 2$ and $\delta$ satisfying $2 \leq \delta \leq 2^{\mu} - 1$, a *binary BCH code* is a [cyclic code](/ref/CyclicCode) with generator polynomial given by
    $$
        g(X) = \mathrm{lcm} \left\\{ \phi_1(X), \phi_2(X), \ldots, \phi_{\delta - 1}(X) \right\\},
    $$
    where $\phi_i(X)$ is the minimal polynomial of $\alpha^i$, and $\alpha$ is a primitive element of $\mathrm{GF}(2^\mu)$. The parameter $\delta$ must be a *Bose distance*. The resulting code is denoted by $\bch(\mu, \delta)$, and has the following parameters, where $\delta = 2 \tau + 1$:

    - Length: $n = 2^{\mu} - 1$
    - Dimension: $k \geq n - \mu \tau$
    - Redundancy: $m \leq \mu \tau$
    - Minimum distance: $d \geq \delta$

    Only *narrow-sense* and *primitive* BCH codes are implemented. For more details, see <cite>LC04, Ch. 6</cite> and <cite>HP03, Sec. 5.1</cite>.

    Notes:
        - For $\delta = 3$ it reduces to the [Hamming code](/ref/HammingCode).
        - For $\delta = 2^{\mu} - 1$ it reduces to the [repetition code](/ref/RepetitionCode).

    Parameters:
        mu: The parameter $\mu$ of the BCH code.
        delta: The Bose distance $\delta$ of the BCH code.

    **Resources:**

    - [Table of possible Bose distances.](/res/bch-codes#bose-distances)

    Examples:
        >>> code = komm.BCHCode(mu=5, delta=7)
        >>> (code.length, code.dimension, code.redundancy)
        (31, 16, 15)
        >>> code.generator_polynomial
        BinaryPolynomial(0b1000111110101111)
        >>> code.minimum_distance()
        7

        >>> komm.BCHCode(mu=7, delta=31)
        BCHCode(mu=7, delta=31)

        >>> komm.BCHCode(mu=7, delta=32)
        Traceback (most recent call last):
        ...
        ValueError: 'delta' must be a Bose distance (next one is 43)

        >>> komm.BCHCode(mu=7, delta=43)
        BCHCode(mu=7, delta=43)
    """

    def __init__(self, mu: int, delta: int) -> None:
        if not mu >= 2:
            raise ValueError("'mu' must satisfy mu >= 2")
        if not 2 <= delta <= 2**mu - 1:
            raise ValueError("'delta' must satisfy 2 <= delta <= 2**mu - 1")

        field = FiniteBifield(mu)
        # Since the default modulus is a primitive polynomial, alpha = X is a primitive element.
        alpha = field(0b10)

        def phi(i: int):
            return (alpha**i).minimal_polynomial()

        lcm_set = {phi(i) for i in range(1, delta)}

        if phi(delta) in lcm_set:
            bose = next(d for d in range(delta + 1, 2**mu) if phi(d) not in lcm_set)
            raise ValueError(f"'delta' must be a Bose distance (next one is {bose})")

        self.mu = mu
        self.delta = delta
        self.field = field
        self.alpha = alpha

        super().__init__(
            length=2**mu - 1,
            generator_polynomial=reduce(operator.mul, lcm_set),
        )

    def __repr__(self) -> str:
        args = f"mu={self.mu}, delta={self.delta}"
        return f"{self.__class__.__name__}({args})"

    def bch_syndrome(
        self, r_poly: BinaryPolynomial
    ) -> list[FiniteBifieldElement[FiniteBifield]]:
        # BCH syndrome computation. See [LC04, p. 205–209].
        return [r_poly.evaluate(self.alpha**i) for i in range(1, self.delta)]
