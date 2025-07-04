import operator
from functools import reduce

import pytest
from typeguard import TypeCheckError

import komm


def test_bch_generator_polynomial():
    # [LC04, Table 6.4]
    mu = 6
    factors = {
        1: komm.BinaryPolynomial(0b1000011),
        2: komm.BinaryPolynomial(0b1010111),
        3: komm.BinaryPolynomial(0b1100111),
        4: komm.BinaryPolynomial(0b1001001),
        5: komm.BinaryPolynomial(0b0001101),
        6: komm.BinaryPolynomial(0b1101101),
        7: komm.BinaryPolynomial(0b1011011),
        10: komm.BinaryPolynomial(0b1110101),
        11: komm.BinaryPolynomial(0b0000111),
        13: komm.BinaryPolynomial(0b1110011),
        15: komm.BinaryPolynomial(0b0001011),
    }
    dimensions = {
        1: 57,
        2: 51,
        3: 45,
        4: 39,
        5: 36,
        6: 30,
        7: 24,
        10: 18,
        11: 16,
        13: 10,
        15: 7,
    }

    for tau, dimension in dimensions.items():
        delta = 2 * tau + 1
        code = komm.BCHCode(mu=mu, delta=delta)
        assert code.length == 63
        assert code.dimension == dimension
        assert code.generator_polynomial == reduce(
            operator.mul,
            [factors.get(i, komm.BinaryPolynomial(0b1)) for i in range(1, tau + 1)],
        )


parameters = {
    2: [3],
    3: [3, 7],
    4: [3, 5, 7, 15],
    5: [7, 11, 15, 31],
    6: [23, 27, 31, 63],
    7: [55, 63, 127],
}


def get_mu_delta_pairs():
    pairs = []
    for mu, deltas in parameters.items():
        for delta in deltas:
            pairs.append((mu, delta))
    return pairs


@pytest.mark.parametrize("mu, delta", get_mu_delta_pairs())
def test_bch_minimum_distance(mu, delta):
    code = komm.BCHCode(mu, delta)
    assert code.minimum_distance() >= delta


def test_bch_syndrome():
    # [LC04, Example 6.4]
    code = komm.BCHCode(mu=4, delta=5)
    alpha = code.alpha
    r = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    r_poly = komm.BinaryPolynomial.from_coefficients(r)
    assert code.bch_syndrome(r_poly) == [alpha**2, alpha**4, alpha**7, alpha**8]


def test_bch_code_invalid_init():
    with pytest.raises(ValueError, match="must satisfy mu >= 2"):
        komm.BCHCode(mu=1, delta=3)
    with pytest.raises(ValueError, match="must satisfy 2 <= delta"):
        komm.BCHCode(mu=3, delta=8)
    with pytest.raises(ValueError, match="must be a Bose distance"):
        komm.BCHCode(mu=3, delta=5)
    with pytest.raises(TypeCheckError):
        komm.BCHCode(mu=3, delta="5")  # type: ignore
