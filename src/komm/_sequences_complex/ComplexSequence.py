import numpy as np

from .._util.correlation import acorr, cyclic_acorr


class ComplexSequence:
    r"""
    General complex sequence. It is denoted by $x[n]$, with elements in $\\mathbb{C}$. Its length (or period) is denoted by $L$.
    """

    def __init__(self, sequence):
        r"""
        Constructor for the class.

        Parameters:
            sequence (Array1D[complex]): The complex sequence. Must be a 1D-array of length $L$ with elements in $\\mathbb{C}$.

        Examples:
            >>> seq = ComplexSequence([1, 1j, -1, -1j])
            >>> seq.sequence
            array([ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j])
        """
        self._sequence = np.array(sequence, dtype=complex)
        self._length = self._sequence.size

    def __repr__(self):
        args = r"sequence={}".format(self._sequence.tolist())
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def sequence(self):
        r"""
        The complex sequence $x[n]$.
        """
        return self._sequence

    @property
    def length(self):
        r"""
        The length (or period) $L$ of the complex sequence.
        """
        return self._length

    def autocorrelation(self, shifts=None, normalized=False):
        r"""
        Returns the autocorrelation $R[\ell]$ of the complex sequence. See [`komm.autocorrelation`](/ref/autocorrelation) for more details.

        Parameters:
            shifts (Optional[Array1D[int]]): See [`komm.autocorrelation`](/ref/autocorrelation). The default value yields $[0 : L)$.

            normalized (Optional[bool]): See [`komm.autocorrelation`](/ref/autocorrelation). The default value is `False`.

        Returns:
            autocorrelation (Array1D[complex]): The autocorrelation $R[\ell]$ of the complex sequence.

        Examples:
            >>> seq = ComplexSequence([1, 1j, -1, -1j])
            >>> seq.autocorrelation(shifts=[-2, -1, 0, 1, 2])
            array([-2.+0.j,  0.-3.j,  4.+0.j,  0.+3.j, -2.+0.j])
        """
        return acorr(self._sequence, shifts, normalized)

    def cyclic_autocorrelation(self, shifts=None, normalized=False):
        r"""
        Returns the cyclic autocorrelation $\tilde{R}[\ell]$ of the complex sequence. See [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation) for more details.

        Parameters:
            shifts (Optional[Array1D[int]]): See [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation). The default value yields $[0 : L)$.

            normalized (Optional[bool]): See [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation). The default value is `False`.

        Returns:
            cyclic_autocorrelation (Array1D[complex]): The cyclic autocorrelation $\tilde{R}[\ell]$ of the complex sequence.

        Examples:
            >>> seq = ComplexSequence([1, 1j, -1, -1j])
            >>> seq.cyclic_autocorrelation(shifts=[-2, -1, 0, 1, 2])
            array([-4.+0.j,  0.-4.j,  4.+0.j,  0.+4.j, -4.+0.j])
        """
        return cyclic_acorr(self._sequence, shifts, normalized)
