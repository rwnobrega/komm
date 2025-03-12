from functools import cached_property
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from . import base

T = TypeVar("T", np.floating, np.complexfloating)


class Modulation(base.Modulation[T]):
    r"""
    General modulation scheme. A *modulation scheme* of *order* $M = 2^m$ is defined by a *constellation* $\mathbf{X}$, which is a real or complex vector of length $M$, and a *binary labeling* $\mathbf{Q}$, which is an $M \times m$ binary matrix whose rows are all distinct. The $i$-th element of $\mathbf{X}$, for $i \in [0:M)$, is denoted by $x_i$ and is called the $i$-th *constellation symbol*. The $i$-th row of $\mathbf{Q}$, for $i \in [0:M)$, is called the *binary representation* of the $i$-th constellation symbol. For more details, see <cite>SA15, Sec. 2.5</cite>."

    Parameters:
        constellation: The constellation $\mathbf{X}$ of the modulation. Must be a 1D-array containing $M$ real or complex numbers.

        labeling: The binary labeling $\mathbf{Q}$ of the modulation. Must be a 2D-array of shape $(M, m)$ where each row is a distinct binary $m$-tuple.

    Examples:
        1. The real modulation scheme depicted in the figure below has $M = 4$ and $m = 2$.
            <figure markdown>
            ![Example for real modulation with M = 4](/figures/modulation_real_4.svg)
            </figure>
            The constellation and labeling are given, respectively, by
            $$
            \mathbf{X} = \begin{bmatrix}
                -0.5 \\\\
                0.0 \\\\
                0.5 \\\\
                2.0
            \end{bmatrix}
            \qquad\text{and}\qquad
            \mathbf{Q} = \begin{bmatrix}
                1 & 0 \\\\
                1 & 1 \\\\
                0 & 1 \\\\
                0 & 0
            \end{bmatrix}.
            $$

                >>> modulation = komm.Modulation(
                ...     constellation=[-0.5, 0.0, 0.5, 2.0],
                ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
                ... )

        1. The complex modulation scheme depicted in the figure below has $M = 4$ and $m = 2$.
            <figure markdown>
            ![Example for complex modulation with M = 4](/figures/modulation_complex_4.svg)
            </figure>
            The constellation and labeling are given, respectively, by
            $$
            \mathbf{X} = \begin{bmatrix}
                0 \\\\
                -1 \\\\
                1 \\\\
                \mathrm{j}
            \end{bmatrix}
            \qquad\text{and}\qquad
            \mathbf{Q} = \begin{bmatrix}
                0 & 0 \\\\
                0 & 1 \\\\
                1 & 0 \\\\
                1 & 1
            \end{bmatrix}.
            $$

                >>> modulation = komm.Modulation(
                ...     constellation=[0, -1, 1, 1j],
                ...     labeling=[[0, 0], [0, 1], [1, 0], [1, 1]],
                ... )
    """

    def __init__(self, constellation: npt.ArrayLike, labeling: npt.ArrayLike) -> None:
        self._constellation = np.asarray(constellation)
        self._labeling = np.asarray(labeling)
        super()._validate_parameters()

    def __repr__(self) -> str:
        args = ", ".join([
            f"constellation={self.constellation.tolist()}",
            f"labeling={self.labeling.tolist()}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def constellation(self) -> npt.NDArray[T]:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.constellation
            array([-0.5,  0. ,  0.5,  2. ])
        """
        return self._constellation

    @cached_property
    def labeling(self) -> npt.NDArray[np.integer]:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.labeling
            array([[1, 0],
                   [1, 1],
                   [0, 1],
                   [0, 0]])
        """
        return self._labeling

    @cached_property
    def inverse_labeling(self) -> dict[tuple[int, ...], int]:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.inverse_labeling
            {(1, 0): 0, (1, 1): 1, (0, 1): 2, (0, 0): 3}
        """
        return super().inverse_labeling

    @cached_property
    def order(self) -> int:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.order
            4
        """
        return super().order

    @cached_property
    def bits_per_symbol(self) -> int:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.bits_per_symbol
            2
        """
        return super().bits_per_symbol

    @cached_property
    def energy_per_symbol(self) -> float:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.energy_per_symbol
            1.125

            >>> modulation = komm.Modulation(
            ...     constellation=[0, -1, 1, 1j],
            ...     labeling=[[0, 0], [0, 1], [1, 0], [1, 1]],
            ... )
            >>> modulation.energy_per_symbol
            0.75
        """
        return super().energy_per_symbol

    @cached_property
    def energy_per_bit(self) -> float:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.energy_per_bit
            0.5625

            >>> modulation = komm.Modulation(
            ...     constellation=[0, -1, 1, 1j],
            ...     labeling=[[0, 0], [0, 1], [1, 0], [1, 1]],
            ... )
            >>> modulation.energy_per_bit
            0.375
        """
        return super().energy_per_bit

    @cached_property
    def symbol_mean(self) -> float | complex:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.symbol_mean
            0.5

            >>> modulation = komm.Modulation(
            ...     constellation=[0, -1, 1, 1j],
            ...     labeling=[[0, 0], [0, 1], [1, 0], [1, 1]],
            ... )
            >>> modulation.symbol_mean
            0.25j
        """
        return super().symbol_mean

    @cached_property
    def minimum_distance(self) -> float:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.minimum_distance
            0.5

            >>> modulation = komm.Modulation(
            ...     constellation=[0, -1, 1, 1j],
            ...     labeling=[[0, 0], [0, 1], [1, 0], [1, 1]],
            ... )
            >>> modulation.minimum_distance
            1.0
        """
        return super().minimum_distance

    def modulate(self, input: npt.ArrayLike) -> npt.NDArray[T]:
        r"""
        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.modulate([0, 0, 1, 1, 0, 0, 1, 0])
            array([ 2. ,  0. ,  2. , -0.5])
            >>> modulation.modulate([[0, 0, 1, 1], [0, 0, 1, 0]])
            array([[ 2. ,  0. ],
                   [ 2. , -0.5]])
        """
        return super().modulate(input)

    def demodulate_hard(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Note:
            This method implements the general minimum Euclidean distance hard demodulator, assuming uniformly distributed symbols.

        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.demodulate_hard([2.17, -0.06, 1.94, -0.61])
            array([0, 0, 1, 1, 0, 0, 1, 0])
            >>> modulation.demodulate_hard([[2.17, -0.06], [1.94, -0.61]])
            array([[0, 0, 1, 1],
                   [0, 0, 1, 0]])
        """
        return super().demodulate_hard(input)

    def demodulate_soft(
        self, input: npt.ArrayLike, snr: float = 1.0
    ) -> npt.NDArray[np.floating]:
        r"""
        Note:
            This method implements the general soft demodulator, assuming uniformly distributed symbols.

        Examples:
            >>> modulation = komm.Modulation(
            ...     constellation=[-0.5, 0.0, 0.5, 2.0],
            ...     labeling=[[1, 0], [1, 1], [0, 1], [0, 0]],
            ... )
            >>> modulation.demodulate_soft([2.17, -0.06, 1.94, -0.61], snr=100.0).round(1)
            array([ 416. ,  245.3,  -27.6,  -16.9,  334.2,  184. , -108.4,   32. ])
            >>> modulation.demodulate_soft([[2.17, -0.06], [1.94, -0.61]], snr=100.0).round(1)
            array([[ 416. ,  245.3,  -27.6,  -16.9],
                   [ 334.2,  184. , -108.4,   32. ]])
        """
        return super().demodulate_soft(input, snr)
