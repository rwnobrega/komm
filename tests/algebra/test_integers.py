import komm
from komm._algebra.domain import DomainElement
from komm._algebra.Integers import Integers
from komm._algebra.ring import Ring, RingElement


def test_integer_protocol():
    assert isinstance(Integers, Ring)

    integer = komm.Integer(42)
    assert isinstance(integer.ambient, Integers)
    assert isinstance(integer, RingElement)
    assert isinstance(integer, DomainElement)
