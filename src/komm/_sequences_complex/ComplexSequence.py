from typing import Optional

import numpy as np
import numpy.typing as npt

from .._util.correlation import acorr, cyclic_acorr


class ComplexSequence:
    r"""
    General complex sequence. It is denoted by $x[n]$, with elements in $\\mathbb{C}$. Its length (or period) is denoted by $L$.

    Parameters:
        sequence (Array1D[complex]): The complex sequence. Must be a 1D-array of length $L$ with elements in $\\mathbb{C}$.

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

    @property
    def length(self):
        r"""
        The length (or period) $L$ of the complex sequence.
        """
        return self.sequence.size

    def autocorrelation(
        self, shifts: Optional[npt.ArrayLike] = None, normalized: bool = False
    ) -> npt.NDArray[np.complex128]:
        r"""
        Returns the autocorrelation $R[\ell]$ of the complex sequence. See [`komm.acorr`](/ref/acorr) for more details.

        Parameters:
            shifts: See the corresponding parameter in [`komm.acorr`](/ref/acorr).
            normalized: See the corresponding parameter in [`komm.acorr`](/ref/acorr).

        Returns:
            The autocorrelation $R[\ell]$ of the complex sequence.

        Examples:
            >>> seq = ComplexSequence([1, 1j, -1, -1j])
            >>> seq.autocorrelation(shifts=[-2, -1, 0, 1, 2])
            array([-2.+0.j,  0.-3.j,  4.+0.j,  0.+3.j, -2.+0.j])
        """
        return acorr(self.sequence, shifts, normalized)

    def cyclic_autocorrelation(
        self, shifts: Optional[npt.ArrayLike] = None, normalized: bool = False
    ) -> npt.NDArray[np.complex128]:
        r"""
        Returns the cyclic autocorrelation $\tilde{R}[\ell]$ of the complex sequence. See [`komm.cyclic_acorr`](/ref/cyclic_acorr) for more details.

        Parameters:
            shifts: See the corresponding parameter in [`komm.cyclic_acorr`](/ref/cyclic_acorr).
            normalized: See the corresponding parameter in [`komm.cyclic_acorr`](/ref/cyclic_acorr).

        Returns:
            The cyclic autocorrelation $\tilde{R}[\ell]$ of the complex sequence.

        Examples:
            >>> seq = ComplexSequence([1, 1j, -1, -1j])
            >>> seq.cyclic_autocorrelation(shifts=[-2, -1, 0, 1, 2])
            array([-4.+0.j,  0.-4.j,  4.+0.j,  0.+4.j, -4.+0.j])
        """
        return cyclic_acorr(self.sequence, shifts, normalized)
