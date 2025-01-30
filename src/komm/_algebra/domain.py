from typing import Protocol, TypeVar, runtime_checkable

from typing_extensions import Self

from . import ring

T = TypeVar("T", bound="DomainElement")


@runtime_checkable
class DomainElement(ring.RingElement, Protocol):
    def __floordiv__(self: Self, other: Self) -> Self: ...
    def __mod__(self: Self, other: Self) -> Self: ...


def gcd(x: T, y: T) -> T:
    r"""
    Greatest common divisor. Computes the greatest common divisor (gcd) of two elements in a domain using the Euclidean algorithm.

    Parameters:
        x: First element
        y: Second element

    Returns:
        gcd: The greatest common divisor of `x` and `y`

    References:
        `Euclidean algorithm <https://en.wikipedia.org/wiki/Euclidean_algorithm>`_
    """
    zero = x.ambient.zero
    while y != zero:
        x, y = y, x % y
    return x


def xgcd(x: T, y: T) -> tuple[T, T, T]:
    r"""
    Extended gcd. Computes the greatest common divisor (gcd) and the Bézout coefficients of two elements in a domain using the extended Euclidean algorithm.

    Parameters:
        x: First element
        y: Second element

    Returns:
        gcd: The greatest common divisor of `x` and `y`
        s: The Bézout coefficient of `x`
        t: The Bézout coefficient of `y`

    References:
        `Extended Euclidean algorithm <https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm>`_
    """
    zero: T = x.ambient.zero
    one: T = x.ambient.one
    if x == zero:
        return y, zero, one
    else:
        d, s, t = xgcd(y % x, x)
        return d, t - s * (y // x), s
