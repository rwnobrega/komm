from attrs import field, frozen
from typing_extensions import Self


@frozen
class Integer:
    value: int = field(converter=int)

    @property
    def ambient(self) -> "Integers":
        return Integers()

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

    @property
    def zero(self) -> Integer:
        return Integer(0)

    @property
    def one(self) -> Integer:
        return Integer(1)
