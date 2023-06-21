import numpy as np

from .._util import acorr, cyclic_acorr


class BinarySequence:
    r"""
    General binary sequence. It may be represented either in *bit format*, denoted by $b[n]$, with elements in the set $\\{ 0, 1 \\}$, or in *polar format*, denoted by $x[n]$, with elements in the set $\\{ \pm 1 \\}$. The correspondences $0 \mapsto +1$ and $1 \mapsto -1$ from bit format to polar format is assumed.
    """

    def __init__(self, bit_sequence=None, polar_sequence=None):
        r"""
        Constructor for the class. It expects *exactly one* the following parameters:

        Parameters:

            bit_sequence (Array1D[int]): The binary sequence in bit format. Must be a 1D-array with elements in $\\{ 0, 1 \\}$.

            polar_sequence (Array1D[int]): The binary sequence in polar format. Must be a 1D-array with elements in $\\{ \pm 1 \\}$.

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
        if bit_sequence is not None and polar_sequence is None:
            self._bit_sequence = np.array(bit_sequence, dtype=int)
            self._polar_sequence = (-1) ** self._bit_sequence
            self._constructed_from = "bit_sequence"
        elif polar_sequence is not None and bit_sequence is None:
            self._polar_sequence = np.array(polar_sequence, dtype=int)
            self._bit_sequence = 1 * (self._polar_sequence < 0)
            self._constructed_from = "polar_sequence"
        else:
            raise ValueError("Either specify 'bit_sequence' or 'polar_sequence'")
        self._length = self._bit_sequence.size

    def __repr__(self):
        args = "bit_sequence={}".format(self._bit_sequence.tolist())
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def bit_sequence(self):
        r"""
        The binary sequence in bit format, $b[n] \in \\{ 0, 1 \\}$.
        """
        return self._bit_sequence

    @property
    def polar_sequence(self):
        r"""
        The binary sequence in polar format, $x[n] \in \\{ \pm 1 \\}$.
        """
        return self._polar_sequence

    @property
    def length(self):
        r"""
        The length (or period) $L$ of the binary sequence.
        """
        return self._length

    def autocorrelation(self, shifts=None, normalized=False):
        r"""
        Returns the autocorrelation $R[\ell]$ of the binary sequence in polar format. See [`komm.autocorrelation`](/ref/autocorrelation) for more details.

        Parameters:

            shifts (Optional[Array1D[int]]): See [`komm.autocorrelation`](/ref/autocorrelation). The default value yields $[0 : L)$.

            normalized (Optional[bool]): See [`komm.autocorrelation`](/ref/autocorrelation). The default value is `False`.

        Returns:

            autocorrelation (Array1D[complex]): The autocorrelation $R[\ell]$ of the complex sequence.

        Examples:

            >>> seq = komm.BinarySequence(bit_sequence=[0, 1, 1, 0])
            >>> seq.autocorrelation()
            array([ 4, -1, -2,  1])
        """
        return acorr(self._polar_sequence, shifts=shifts, normalized=normalized)

    def cyclic_autocorrelation(self, shifts=None, normalized=False):
        r"""
        Returns the cyclic autocorrelation $\tilde{R}[\ell]$ of the binary sequence in polar format. See [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation) for more details.

        Parameters:

            shifts (Optional[Array1D[int]]): See [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation). The default value yields $[0 : L)$.

            normalized (Optional[bool]): See [`komm.cyclic_autocorrelation`](/ref/cyclic_autocorrelation). The default value is `False`.

        Returns:

            cyclic_autocorrelation (Array1D[complex]): The cyclic autocorrelation $\tilde{R}[\ell]$ of the complex sequence.

        Examples:

            >>> seq = komm.BinarySequence(bit_sequence=[0, 1, 1, 0])
            >>> seq.cyclic_autocorrelation()
            array([ 4,  0, -4,  0])
        """
        return cyclic_acorr(self._polar_sequence, shifts=shifts, normalized=normalized)
