import numpy as np
import numpy.typing as npt

from .. import abc
from .._util.validators import validate_integer_min, validate_integer_range


class UnaryCode(abc.IntegerCode):
    r"""
    Unary code. It is an integer code. The codeword for a non-negative integer $n$ consists of $n - 1$ zeros followed by a single $1$. For more details, see <cite>MacK03, Ch. 7</cite>.
    """

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        For the unary code, the integers must be positive.

        Examples:
            >>> code = komm.UnaryCode()
            >>> code.encode([4, 1, 3])
            array([0, 0, 0, 1, 1, 0, 0, 1])
        """
        input = validate_integer_min(input, low=1)
        if input.size == 0:
            return np.array([], dtype=int)
        return np.concatenate([unary_encode(i) for i in input])

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> code.decode([0, 0, 0, 1, 1, 0, 0, 1])
            array([4, 1, 3])
        """
        input = validate_integer_range(input, low=0, high=2)
        output: list[int] = []
        i = 0
        while i < input.size:
            j = i
            while j < input.size and input[j] == 0:
                j += 1
            if j == input.size:
                raise ValueError("input contains an incomplete codeword")
            output.append(unary_decode(list(input[i : j + 1])))
            i = j + 1
        return np.array(output)


def unary_encode(integer: int) -> list[int]:
    return [0] * (integer - 1) + [1]


def unary_decode(bits: list[int]) -> int:
    return len(bits)
