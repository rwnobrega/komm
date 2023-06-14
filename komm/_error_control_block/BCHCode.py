import functools
import operator

import numpy as np

from .._algebra import BinaryPolynomial, FiniteBifield
from .._aux import tag
from .CyclicCode import CyclicCode


class BCHCode(CyclicCode):
    r"""
    Bose–Chaudhuri–Hocquenghem (BCH) code. It is a [cyclic code](/ref/CyclicCode) specified by two integers $\mu$ and $\tau$ which must satisfy $1 \leq \tau < 2^{\mu - 1}$. The parameter $\tau$ is called the *designed error-correcting capability* of the BCH code; it will be internally replaced by the true error-correcting capability $t$ of the code. See references for more details. The resulting code is denoted by $\bch(\mu, \tau)$, and has the following parameters:

    - Length: $n = 2^{\mu} - 1$
    - Dimension: $k \geq n - \mu \tau$
    - Redundancy: $m \leq \mu \tau$
    - Minimum distance: $d \geq 2\tau + 1$

    For more details, see <cite>LC04, Ch. 6</cite>.
    """

    def __init__(self, mu, tau):
        r"""
        Constructor for the class.

        Parameters:

            mu (int): The parameter $\mu$ of the code.

            tau (int): The designed error-correcting capability $\tau$ of the BCH code. It will be internally replaced by the true error-correcting capability $t$ of the code.

        Examples:

            >>> code = komm.BCHCode(5, 3)
            >>> (code.length, code.dimension, code.minimum_distance)
            (31, 16, 7)
            >>> code.generator_polynomial
            BinaryPolynomial(0b1000111110101111)

            >>> # The true error-correcting capability is equal to the designed one:
            >>> code = komm.BCHCode(7, 15); code
            BCHCode(7, 15)
            >>> # The true error-correcting capability is greater than the designed one:
            >>> code = komm.BCHCode(7, 16); code
            BCHCode(7, 21)
        """
        if not 1 <= tau < 2 ** (mu - 1):
            raise ValueError("Parameters must satisfy 1 <= tau < 2**(mu - 1)")

        field = FiniteBifield(mu)
        generator_polynomial, t = self._bch_code_generator_polynomial(field, tau)
        super().__init__(length=2**mu - 1, generator_polynomial=generator_polynomial)

        self._field = field
        self._mu = mu
        self._packing_radius = t
        self._minimum_distance = 2 * t + 1

        alpha = field.primitive_element
        self._beta = [alpha ** (i + 1) for i in range(2 * t)]
        self._beta_minimal_polynomial = [b.minimal_polynomial() for b in self._beta]

    def __repr__(self):
        args = "{}, {}".format(self._mu, self._packing_radius)
        return "{}({})".format(self.__class__.__name__, args)

    @staticmethod
    def _bch_code_generator_polynomial(field, tau):
        r"""
        Assumes 1 <= tau < 2**(mu - 1). See <cite>LC04, p. 194–195</cite>.
        """
        alpha = field.primitive_element

        t = tau
        lcm_set = {(alpha ** (2 * i + 1)).minimal_polynomial() for i in range(t)}
        while True:
            if (alpha ** (2 * t + 1)).minimal_polynomial() not in lcm_set:
                break
            t += 1
        generator_polynomial = functools.reduce(operator.mul, lcm_set)

        return generator_polynomial, t

    def _bch_general_decoder(self, recvword, syndrome_computer, key_equation_solver, root_finder):
        r"""
        General BCH decoder. See <cite>LC04, p. 205–209</cite>.
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
        r"""
        BCH syndrome computation. See <cite>LC04, p. 205–209</cite>.
        """
        syndrome_polynomial = np.empty(len(self._beta), dtype=object)
        for i, (b, b_min_polynomial) in enumerate(zip(self._beta, self._beta_minimal_polynomial)):
            syndrome_polynomial[i] = (recvword_polynomial % b_min_polynomial).evaluate(b)
        return syndrome_polynomial

    def _find_roots(self, polynomial):
        r"""
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
        r"""
        Berlekamp's iterative procedure for finding the error-location polynomial of a BCH code. See <cite>LC04, p. 209–212</cite> and <cite>RL09, p. 114–121</cite>.
        """
        field = self._field
        t = self._packing_radius

        sigma = {-1: np.array([field(1)], dtype=object), 0: np.array([field(1)], dtype=object)}
        discrepancy = {-1: field(1), 0: syndrome_polynomial[0]}
        degree = {-1: 0, 0: 0}

        # TODO: This mu is not the same as the mu in __init__...
        for mu in range(2 * t):
            if discrepancy[mu] == field(0):
                degree[mu + 1] = degree[mu]
                sigma[mu + 1] = sigma[mu]
            else:
                rho, max_so_far = -1, -1
                for i in range(-1, mu):
                    if discrepancy[i] != field(0) and i - degree[i] > max_so_far:
                        rho, max_so_far = i, i - degree[i]
                degree[mu + 1] = max(degree[mu], degree[rho] + mu - rho)
                sigma[mu + 1] = np.array([field(0)] * (degree[mu + 1] + 1), dtype=object)
                first_guy = np.array([field(0)] * (degree[mu + 1] + 1), dtype=object)
                first_guy[: degree[mu] + 1] = sigma[mu]
                second_guy = np.array([field(0)] * (degree[mu + 1] + 1), dtype=object)
                second_guy[mu - rho : degree[rho] + mu - rho + 1] = sigma[rho]
                e = discrepancy[mu] / discrepancy[rho]
                second_guy = np.array([e * x for x in second_guy], dtype=object)
                sigma[mu + 1] = first_guy + second_guy
            if mu < 2 * t - 1:
                discrepancy[mu + 1] = syndrome_polynomial[mu + 1]
                for idx in range(1, degree[mu + 1] + 1):
                    discrepancy[mu + 1] += sigma[mu + 1][idx] * syndrome_polynomial[mu + 1 - idx]

        return sigma[2 * t]

    @tag(name="Berlekamp decoder", input_type="hard", target="codeword")
    def _decode_berlekamp(self, recvword):
        return self._bch_general_decoder(
            recvword,
            syndrome_computer=self._bch_syndrome,
            key_equation_solver=self._berlekamp_algorithm,
            root_finder=self._find_roots,
        )

    def _default_decoder(self, dtype):
        if dtype == int:
            return "berlekamp"
        else:
            return super()._default_decoder(dtype)
