from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from .._error_control_block import PolarCode
from .._util.decorators import blockwise, vectorize
from .._util.special_functions import boxplus
from . import base

Belief: TypeAlias = npt.NDArray[np.floating]
Decision: TypeAlias = npt.NDArray[np.integer]
Node: TypeAlias = tuple[int, int]  # depth, index


def g(r: Belief, s: Belief, b: Decision) -> Belief:
    return s + (-1) ** b * r


@dataclass
class SCDecoder(base.BlockDecoder[PolarCode]):
    r"""
    Successive cancellation decoder for [Polar codes](/ref/PolarCode).

    Parameters:
        code: The Polar code to be used for decoding.
        output_type: The type of the output. Either `'hard'` or `'soft'`. Default is `'soft'`.

    Notes:
        - Input type: `soft` (L-values).
        - Output type: `hard` (bits) or `soft` (L-values).
    """

    code: PolarCode
    output_type: Literal["hard", "soft"] = "soft"

    def __post_init__(self) -> None:
        self._f = boxplus

    def __call__(self, input: npt.ArrayLike) -> npt.NDArray[np.integer | np.floating]:
        r"""
        Examples:
            >>> code = komm.PolarCode(3, [0, 1, 2, 4])

            >>> decoder = komm.SCDecoder(code)
            >>> decoder([1, -4, -3, 2, -2, 3, 4, -1])
            array([ -6.84595089,  -5.96379094,  -9.30685282, -20.        ])

            >>> decoder = komm.SCDecoder(code, output_type="hard")
            >>> decoder([1, -4, -3, 2, -2, 3, 4, -1])
            array([1, 1, 1, 1])
        """

        @blockwise(self.code.length)
        @vectorize
        def decode(li: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            # See [Successive Cancellation(SC) Decoder for a General (N,K) Polar Code]
            # by Prof. Andrew Thangaraj (NPTEL-NOC IITM)
            active = (0, 0)
            beliefs: dict[Node, Belief] = {(0, 0): li}
            decisions: dict[Node, Decision] = {}
            while active[0] >= 0:
                depth, index = active
                parent = (depth - 1, index // 2)
                if depth == self.code.mu:  # Leaf node
                    if index in self.code.frozen:
                        decisions[active] = np.array([0])
                    else:
                        decisions[active] = (beliefs[active] < 0).astype(int)
                    active = parent
                    continue
                # Interior node
                child_l = (depth + 1, index * 2)
                child_r = (depth + 1, index * 2 + 1)
                msg = beliefs[active]
                M = msg.size
                if child_l not in decisions:  # Step L:
                    beliefs[child_l] = self._f(msg[: M // 2], msg[M // 2 :])
                    active = child_l
                elif child_r not in decisions:  # Step R:
                    side_msg = decisions[child_l]
                    beliefs[child_r] = g(msg[: M // 2], msg[M // 2 :], side_msg)
                    active = child_r
                else:  # Step U:
                    a = decisions[child_l]
                    b = decisions[child_r]
                    decisions[active] = np.concatenate([a ^ b, b])
                    active = parent
            lo = np.array([beliefs[self.code.mu, i][0] for i in self.code.active])
            return lo

        output = decode(input)
        if self.output_type == "hard":
            return (output < 0).astype(int)
        else:
            return output
