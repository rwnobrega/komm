import numpy as np
import numpy.typing as npt

from . import base


class UnaryCode(base.IntegerCode):
    r"""
    Unary code. It is an integer code. For the definition of this code, see [Wikipedia: Unary coding](https://en.wikipedia.org/wiki/Unary_coding).
    """

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> code = komm.UnaryCode()
            >>> code.encode([4, 1, 3])
            array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0])
        """
        input = np.asarray(input)
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
        input = np.asarray(input)
        output: list[int] = []
        i = 0
        while i < input.size:
            j = i
            while input[j] == 1:
                j += 1
            output.append(unary_decode(list(input[i : j + 1])))
            i = j + 1
        return np.array(output)


def unary_encode(integer: int) -> list[int]:
    return [1] * integer + [0]


def unary_decode(bits: list[int]) -> int:
    return len(bits) - 1
