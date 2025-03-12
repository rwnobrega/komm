from functools import cached_property

import numpy as np
import numpy.typing as npt

from .._util.correlation import autocorrelation, cyclic_autocorrelation


class BinarySequence:
    r"""
    General binary sequence. It may be represented either in *bit format*, denoted by $b[n]$, with elements in the set $\\{ 0, 1 \\}$, or in *polar format*, denoted by $x[n]$, with elements in the set $\\{ \pm 1 \\}$. The correspondences $0 \mapsto +1$ and $1 \mapsto -1$ from bit format to polar format is assumed.

    The constructor expects either the bit sequence or the polar sequence.

    Parameters:
        bit_sequence: The binary sequence in bit format, $b[n] \in \\{ 0, 1 \\}$.
        polar_sequence: The binary sequence in polar format, $x[n] \in \\{ \pm 1 \\}$.

    Examples:
        >>> seq = komm.BinarySequence(bit_sequence=[0, 1, 1, 0])
        >>> seq.bit_sequence
        array([0, 1, 1, 0])
        >>> seq.polar_sequence
        array([ 1, -1, -1,  1])

        >>> seq = komm.BinarySequence(polar_sequence=[1, -1, -1, 1])
        >>> seq.bit_sequence
        array([0, 1, 1, 0])
        >>> seq.polar_sequence
        array([ 1, -1, -1,  1])
    """

    def __init__(
        self,
        bit_sequence: npt.ArrayLike | None = None,
        polar_sequence: npt.ArrayLike | None = None,
    ) -> None:
        if bit_sequence is not None and polar_sequence is None:
            self.bit_sequence = np.asarray(bit_sequence, dtype=int)
            self.polar_sequence = (-1) ** self.bit_sequence
        elif polar_sequence is not None and bit_sequence is None:
            self.polar_sequence = np.asarray(polar_sequence, dtype=int)
            self.bit_sequence = 1 * (self.polar_sequence < 0)
        else:
            raise ValueError("either specify 'bit_sequence' or 'polar_sequence'")

    def __repr__(self) -> str:
        args = f"bit_sequence={self.bit_sequence.tolist()}"
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def length(self) -> int:
        r"""
        The length (or period) $L$ of the binary sequence.
        """
        return self.bit_sequence.size

    def autocorrelation(
        self, shifts: npt.ArrayLike | None = None, normalized: bool = False
    ) -> npt.NDArray[np.floating]:
        r"""
        Returns the autocorrelation $R[\ell]$ of the binary sequence in polar format. See [`komm.autocorrelation`](/ref/autocorrelation) for more details.

        Parameters:
            shifts: See the corresponding parameter in [`komm.autocorrelation`](/ref/autocorrelation).
            normalized: See the corresponding parameter in [`komm.autocorrelation`](/ref/autocorrelation).

        Returns:
            The autocorrelation $R[\ell]$ of the binary sequence.

        Examples:
            >>> seq = komm.BinarySequence(bit_sequence=[0, 1, 1, 0])
            >>> seq.autocorrelation()
            array([ 4, -1, -2,  1])
        """
        return autocorrelation(
            self.polar_sequence, shifts=shifts, normalized=normalized
        )

    def cyclic_autocorrelation(
        self, shifts: npt.ArrayLike | None = None, normalized: bool = False
    ) -> npt.NDArray[np.floating]:
        r"""
        Returns the cyclic autocorrelation $\tilde{R}[\ell]$ of the binary sequence in polar format. See [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation) for more details.

        Parameters:
            shifts: See the corresponding parameter in [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation).
            normalized: See the corresponding parameter in [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation).

        Returns:
            The cyclic autocorrelation $\tilde{R}[\ell]$ of the binary sequence.

        Examples:
            >>> seq = komm.BinarySequence(bit_sequence=[0, 1, 1, 0])
            >>> seq.cyclic_autocorrelation()
            array([ 4,  0, -4,  0])
        """
        return cyclic_autocorrelation(
            self.polar_sequence, shifts=shifts, normalized=normalized
        )
