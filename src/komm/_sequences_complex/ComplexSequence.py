from functools import cached_property

import numpy as np
import numpy.typing as npt

from .._util.correlation import autocorrelation, cyclic_autocorrelation


class ComplexSequence:
    r"""
    General complex sequence. It is denoted by $x[n]$, with elements in $\\mathbb{C}$. Its length (or period) is denoted by $L$.

    Parameters:
        sequence: The complex sequence. Must be a 1D-array of length $L$ with elements in $\\mathbb{C}$.

    Examples:
        >>> seq = ComplexSequence([1, 1j, -1, -1j])
        >>> seq.sequence
        array([ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j])
    """

    def __init__(self, sequence: npt.ArrayLike) -> None:
        self.sequence = np.asarray(sequence, dtype=complex)

    def __repr__(self) -> str:
        args = f"sequence={self.sequence.tolist()}"
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def length(self):
        r"""
        The length (or period) $L$ of the complex sequence.
        """
        return self.sequence.size

    def autocorrelation(
        self, shifts: npt.ArrayLike | None = None, normalized: bool = False
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Returns the autocorrelation $R[\ell]$ of the complex sequence. See [`komm.autocorrelation`](/ref/autocorrelation) for more details.

        Parameters:
            shifts: See the corresponding parameter in [`komm.autocorrelation`](/ref/autocorrelation).
            normalized: See the corresponding parameter in [`komm.autocorrelation`](/ref/autocorrelation).

        Returns:
            The autocorrelation $R[\ell]$ of the complex sequence.

        Examples:
            >>> seq = ComplexSequence([1, 1j, -1, -1j])
            >>> seq.autocorrelation(shifts=[-2, -1, 0, 1, 2])
            array([-2.+0.j,  0.-3.j,  4.+0.j,  0.+3.j, -2.+0.j])
        """
        return autocorrelation(self.sequence, shifts, normalized)

    def cyclic_autocorrelation(
        self, shifts: npt.ArrayLike | None = None, normalized: bool = False
    ) -> npt.NDArray[np.complexfloating]:
        r"""
        Returns the cyclic autocorrelation $\tilde{R}[\ell]$ of the complex sequence. See [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation) for more details.

        Parameters:
            shifts: See the corresponding parameter in [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation).
            normalized: See the corresponding parameter in [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation).

        Returns:
            The cyclic autocorrelation $\tilde{R}[\ell]$ of the complex sequence.

        Examples:
            >>> seq = ComplexSequence([1, 1j, -1, -1j])
            >>> seq.cyclic_autocorrelation(shifts=[-2, -1, 0, 1, 2])
            array([-4.+0.j,  0.-4.j,  4.+0.j,  0.+4.j, -4.+0.j])
        """
        return cyclic_autocorrelation(self.sequence, shifts, normalized)
