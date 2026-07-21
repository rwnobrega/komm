import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.validators import validate_integer_min, validate_integer_range


class UnaryCode(abc.IntegerCode):
    r"""
    Unary code. It is an integer code. For the definition of this code, see [Wikipedia: Unary coding](https://en.wikipedia.org/wiki/Unary_coding).
    """

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encode the input integer array. The integers must be non-negative.

        Examples:
            >>> code = komm.UnaryCode()
            >>> code.encode([4, 1, 3])
            array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0])
        """
        input = validate_integer_min(input, low=0)
        if input.size == 0:
            return np.array([], dtype=int)
        return np.concatenate([unary_encode(i) for i in input])

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> code.decode([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0])
            array([4, 1, 3])
        """
        input = validate_integer_range(input, low=0, high=2)
        output: list[int] = []
        i = 0
        while i < input.size:
            j = i
            while j < input.size and input[j] == 1:
                j += 1
            if j == input.size:
                raise ValueError("input contains an incomplete codeword")
            output.append(unary_decode(list(input[i : j + 1])))
            i = j + 1
        return np.array(output)


def unary_encode(integer: int) -> list[int]:
    return [1] * integer + [0]


def unary_decode(bits: list[int]) -> int:
    return len(bits) - 1
