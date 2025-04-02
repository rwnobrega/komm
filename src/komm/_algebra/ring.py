from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

T_co = TypeVar("T_co", bound="RingElement", covariant=True)


@runtime_checkable
class RingElement(Protocol):
    @property
    def ambient(self) -> "Ring[Any]": ...
    def __add__(self: Self, other: Self) -> Self: ...
    def __sub__(self: Self, other: Self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __mul__(self: Self, other: Self) -> Self: ...
    def __rmul__(self: Self, other: int) -> Self: ...


@runtime_checkable
class Ring(Protocol[T_co]):
    def __call__(self, value: Any) -> T_co: ...
    def __eq__(self, other: Any) -> bool: ...
    @property
    def zero(self) -> T_co: ...
    @property
    def one(self) -> T_co: ...


def power(x: T_co, n: int) -> T_co:
    """Compute $x^n$ using exponentiation by squaring.

    Parameters:
        x: The base (a ring element)
        n: The exponent (a non-negative integer)

    Returns:
        power: The result of `x` raised to the power of `n` in the ring

    Raises:
        ValueError: If `n` is negative

    References:
        `Exponentiation by squaring <https://en.wikipedia.org/wiki/Exponentiation_by_squaring>`_
    """
    if n < 0:
        raise ValueError("Negative exponents not supported")
    if n == 0:
        return x.ambient.one
    if n == 1:
        return x
    y = power(x, n // 2)
    y = y * y
    if n % 2 == 1:
        y = x * y
    return y


def binary_horner(coefficients: npt.ArrayLike, x: T_co) -> T_co:
    r"""
    Specialized Horner's method for binary polynomials. See :func:`horner` for a more general implementation.
    """
    coefficients = np.asarray(coefficients)
    zero: T_co = x.ambient.zero
    one: T_co = x.ambient.one
    result = zero
    for coefficient in reversed(coefficients):
        result *= x
        if coefficient:
            result += coefficient * one
    return result


def horner(coefficients: npt.ArrayLike, x: T_co) -> T_co:
    r"""
    Evaluates a polynomial at a point using Horner's method.

    Parameters:
        poly: Polynomial to evaluate
        x: Point at which to evaluate the polynomial (ring element)

    Returns:
        result: The result of evaluating the polynomial `poly` at point `x`

    References:
        `Horner's method <https://en.wikipedia.org/wiki/Horner's_method>`_
    """
    coefficients = np.asarray(coefficients)
    zero: T_co = x.ambient.zero
    one: T_co = x.ambient.one
    result = zero
    for coefficient in reversed(coefficients):
        result = result * x + coefficient * one
    return result
