from typing import Protocol, Self, TypeVar, runtime_checkable

from . import ring

T = TypeVar("T", bound="DomainElement")


@runtime_checkable
class DomainElement(ring.RingElement, Protocol):
    def __floordiv__(self: Self, other: Self) -> Self: ...
    def __mod__(self: Self, other: Self) -> Self: ...


def gcd(x: T, y: T) -> T:
    r"""
    Greatest common divisor. Computes the greatest common divisor (gcd) of two elements in a domain using the Euclidean algorithm. For more details, see [Wikipedia: Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm).

    Parameters:
        x: First element
        y: Second element

    Returns:
        gcd: The greatest common divisor of `x` and `y`
    """
    zero = x.ambient.zero
    while y != zero:
        x, y = y, x % y
    return x


def xgcd(x: T, y: T) -> tuple[T, T, T]:
    r"""
    Extended gcd. Computes the greatest common divisor (gcd) and the Bézout coefficients of two elements in a domain using the extended Euclidean algorithm. For more details, see [Wikipedia: Extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm).

    Parameters:
        x: First element
        y: Second element

    Returns:
        gcd: The greatest common divisor of `x` and `y`
        s: The Bézout coefficient of `x`
        t: The Bézout coefficient of `y`
    """
    zero: T = x.ambient.zero
    one: T = x.ambient.one
    if x == zero:
        return y, zero, one
    else:
        d, s, t = xgcd(y % x, x)
        return d, t - s * (y // x), s


def lcm(x: T, y: T) -> T:
    r"""
    Least common multiple. Computes the least common multiple (lcm) of two elements in a domain. For more details, see [Wikipedia: Least common multiple](https://en.wikipedia.org/wiki/Least_common_multiple).

    Parameters:
        x: First element
        y: Second element

    Returns:
        lcm: The least common multiple of `x` and `y`
    """
    zero = x.ambient.zero
    if x == zero or y == zero:
        return zero
    g = gcd(x, y)
    return (x // g) * y
