from typing import Protocol, TypeVar, runtime_checkable

from typing_extensions import Self

from . import ring

T_co = TypeVar("T_co", bound="FieldElement", covariant=True)


@runtime_checkable
class FieldElement(ring.RingElement, Protocol):
    def inverse(self: Self) -> Self: ...
    def __truediv__(self: Self, other: Self) -> Self: ...


def power(x: T_co, n: int) -> T_co:
    """Compute $x^n$ using exponentiation by squaring. See the corresponding function in :mod:`komm._algebra.ring`.

    Parameters:
        x: The base (a field element)
        n: The exponent

    Returns:
        power: The result of `x` raised to the power of `n` in the field
    """
    if n < 0:
        return ring.power(x.inverse(), -n)
    else:
        return ring.power(x, n)
