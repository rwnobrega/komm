from typing import Protocol, Self, TypeVar, runtime_checkable

from . import ring

T_co = TypeVar("T_co", bound="FieldElement", covariant=True)


@runtime_checkable
class FieldElement(ring.RingElement, Protocol):
    def inverse(self: Self) -> Self: ...
    def __truediv__(self: Self, other: Self) -> Self: ...


def power(base: T_co, exponent: int) -> T_co:
    r"""
    Computes $b^e$ using exponentiation by squaring. See the corresponding function in :mod:`komm._algebra.ring`.

    Parameters:
        base: The base $b$ (a field element)
        exponent: The exponent $e$

    Returns:
        power: The result of `base` raised to the power of `exponent` in the field
    """
    if exponent < 0:
        return ring.power(base.inverse(), -exponent)
    else:
        return ring.power(base, exponent)
