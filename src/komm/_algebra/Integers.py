from typing_extensions import Self


class Integer:
    def __init__(self, value: int) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def __str__(self) -> str:
        return str(self.value)

    @property
    def ambient(self) -> "Integers":
        return Integers()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value == other.value

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.value + other.value)

    def __sub__(self, other: Self) -> Self:
        return self.__class__(self.value - other.value)

    def __neg__(self) -> Self:
        return self.__class__(-self.value)

    def __mul__(self, other: Self) -> Self:
        return self.__class__(self.value * other.value)

    def __rmul__(self, other: int) -> Self:
        return self.__class__(self.value * other)

    def __divmod__(self, other: Self) -> tuple[Self, Self]:
        quotient, remainder = divmod(self.value, other.value)
        return self.__class__(quotient), self.__class__(remainder)

    def __floordiv__(self, other: Self) -> Self:
        return self.__class__(self.value // other.value)

    def __mod__(self, other: Self) -> Self:
        return self.__class__(self.value % other.value)


class Integers:
    def __call__(self, value: int) -> Integer:
        return Integer(value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def zero(self) -> Integer:
        return Integer(0)

    @property
    def one(self) -> Integer:
        return Integer(1)


def prime_factors(n: int) -> list[int]:
    factors: list[int] = []
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
