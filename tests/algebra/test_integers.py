import komm
from komm._algebra.domain import DomainElement
from komm._algebra.Integers import Integers, prime_factors
from komm._algebra.ring import Ring, RingElement


def test_integer_protocol():
    assert isinstance(Integers, Ring)

    integer = komm.Integer(42)
    assert isinstance(integer.ambient, Integers)
    assert isinstance(integer, RingElement)
    assert isinstance(integer, DomainElement)


def test_prime_factors():
    assert prime_factors(1) == []
    assert prime_factors(2) == [2]
    assert prime_factors(3) == [3]
    assert prime_factors(4) == [2, 2]
    assert prime_factors(5) == [5]
    assert prime_factors(6) == [2, 3]
    assert prime_factors(7) == [7]
    assert prime_factors(8) == [2, 2, 2]
    assert prime_factors(9) == [3, 3]
    assert prime_factors(10) == [2, 5]
    assert prime_factors(11) == [11]
    assert prime_factors(12) == [2, 2, 3]
    assert prime_factors(13) == [13]
    assert prime_factors(14) == [2, 7]
    assert prime_factors(15) == [3, 5]
    assert prime_factors(16) == [2, 2, 2, 2]
    assert prime_factors(17) == [17]
    assert prime_factors(18) == [2, 3, 3]
    assert prime_factors(19) == [19]
    assert prime_factors(20) == [2, 2, 5]
    assert prime_factors(2**31) == [2] * 31
    assert prime_factors(2**31 - 1) == [2_147_483_647]
    assert prime_factors(2**31 + 1) == [3, 715_827_883]
    assert prime_factors(2**32) == [2] * 32
    assert prime_factors(2**32 - 1) == [3, 5, 17, 257, 65_537]
    assert prime_factors(2**32 + 1) == [641, 6_700_417]
