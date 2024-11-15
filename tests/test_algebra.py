import komm


def test_rational_polynomial():
    assert komm.RationalPolynomial([1, 0, -1]) == komm.RationalPolynomial(
        [1, 0, -1, 0, 0, 0]
    )

    poly0 = komm.RationalPolynomial([1])
    poly1 = komm.RationalPolynomial([0, 1])
    assert poly0 + poly1 == poly1 + poly0 == komm.RationalPolynomial([1, 1])

    poly0 = komm.RationalPolynomial([5, -2, 0, 2, 1, 3])
    poly1 = komm.RationalPolynomial([2, 7, 0, 3, 0, 2])
    assert poly0 + poly1 == komm.RationalPolynomial([7, 5, 0, 5, 1, 5])
    assert poly0 - poly1 == komm.RationalPolynomial([3, -9, 0, -1, 1, 1])

    poly0 = komm.RationalPolynomial([5, 0, 0, 0, 0, 2, 3])
    poly1 = komm.RationalPolynomial([2, 5])
    assert poly0 * poly1 == komm.RationalPolynomial([10, 25, 0, 0, 0, 4, 16, 15])


def test_rational_polynomial_divmod():
    poly_dividend = komm.RationalPolynomial([-4, 0, -2, 1])
    poly_divisor = komm.RationalPolynomial([-3, 1])
    poly_quotient = komm.RationalPolynomial([3, 1, 1])
    poly_remainder = komm.RationalPolynomial([5])
    assert divmod(poly_dividend, poly_divisor) == (poly_quotient, poly_remainder)

    poly_dividend = komm.RationalPolynomial([1, 0, 2])
    poly_divisor = komm.RationalPolynomial([0, 0, 0, 0, 0, 1])
    poly_quotient = komm.RationalPolynomial([])
    poly_remainder = komm.RationalPolynomial([1, 0, 2])
    assert divmod(poly_dividend, poly_divisor) == (poly_quotient, poly_remainder)

    poly_dividend = komm.RationalPolynomial([0, 0, 0, 0, 0, 1])
    poly_divisor = komm.RationalPolynomial([1, 0, 2])
    poly_quotient = komm.RationalPolynomial([0, "-1/4", 0, "1/2"])
    poly_remainder = komm.RationalPolynomial([0, "1/4"])
    assert divmod(poly_dividend, poly_divisor) == (poly_quotient, poly_remainder)

    poly_dividend = komm.RationalPolynomial([12, -26, 21, -9, 2])
    poly_divisor = komm.RationalPolynomial([-3, 2])
    poly_quotient = komm.RationalPolynomial([-4, 6, -3, 1])
    poly_remainder = komm.RationalPolynomial([])
    assert poly_dividend // poly_divisor == poly_quotient
    assert poly_dividend % poly_divisor == poly_remainder


def test_rational_polynomial_gcd():
    poly0 = komm.RationalPolynomial([0, 2, 4])
    poly1 = komm.RationalPolynomial([0, 0, 0, 10])
    poly_gcd = komm.RationalPolynomial([0, 1])
    assert komm.RationalPolynomial.gcd(poly0, poly1) == poly_gcd

    poly0 = komm.RationalPolynomial([6, 7, 1])
    poly1 = komm.RationalPolynomial([-6, -5, 1])
    poly_gcd = komm.RationalPolynomial([1, 1])
    assert komm.RationalPolynomial.gcd(poly0, poly1) == poly_gcd


def test_rational_polynomial_fractions():
    fraction = komm.RationalPolynomialFraction([0, 1, 2], [0, 0, 0, 1])
    assert fraction.numerator == komm.RationalPolynomial([1, 2])
    assert fraction.denominator == komm.RationalPolynomial([0, 0, 1])

    fraction = komm.RationalPolynomialFraction([0, "5/14"], [0, 0, 0, "55/21"])
    assert fraction.numerator == komm.RationalPolynomial([3])
    assert fraction.denominator == komm.RationalPolynomial([0, 0, 22])
