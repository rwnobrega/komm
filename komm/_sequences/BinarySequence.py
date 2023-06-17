import functools

import numpy as np
import numpy.typing as npt


class BinarySequence:
    r"""
    General binary sequence. It may be represented either in *bit format*, denoted by $b[n]$, with elements in the set $\\{ 0, 1 \\}$, or in *polar format*, denoted by $a[n]$, with elements in the set $\\{ \pm 1 \\}$. The correspondences $0 \mapsto +1$ and $1 \mapsto -1$ from bit format to polar format is assumed.
    """

    def __init__(self, bit_sequence=None, polar_sequence=None):
        r"""
        Constructor for the class. It expects *exactly one* the following parameters:

        Parameters:

            bit_sequence (Array1D[int]): The binary sequence in bit format. Must be a 1D-array with elements in $\\{ 0, 1 \\}$.

            polar_sequence (Array1D[int]): The binary sequence in polar format. Must be a 1D-array with elements in $\\{ \pm 1 \\}$.
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
        The binary sequence in polar format, $a[n] \in \\{ \pm 1 \\}$.
        """
        return self._polar_sequence

    @property
    def length(self):
        r"""
        The length (or period) $L$ of the binary sequence.
        """
        return self._length

    @functools.cached_property
    def _autocorrelation(self) -> npt.NDArray[np.int_]:
        seq = self._polar_sequence
        L = self._length
        return np.correlate(seq, seq, mode="full")[L - 1 :]

    @functools.cached_property
    def _cyclic_autocorrelation(self) -> npt.NDArray[np.int_]:
        seq = self._polar_sequence
        L = self._length
        return np.array([np.dot(seq, np.roll(seq, ell)) for ell in range(L)])

    def autocorrelation(self, shifts=None, normalized=False):
        r"""
        Returns the autocorrelation $R[\ell]$ of the binary sequence. This is defined as
        $$
            R[\ell] = \sum_{n \in \mathbb{Z}} a[n] a_\ell[n],
        $$
        where $a[n]$ is the binary sequence in polar format, and $a_\ell[n] = a[n - \ell]$ is the sequence $a[n]$ shifted by $\ell$ positions. The autocorrelation $R[\ell]$ is even and satisfies $R[\ell] = 0$ for $|\ell| \geq L$, where $L$ is the length of the binary sequence.

        Parameters:

            shifts (Optional[Array1D[int]]): A 1D-array containing the values of $\ell$ for which the autocorrelation will be computed. The default value is `range(L)`, that is, $[0 : L)$.

            normalized (Optional[bool]): If `True`, returns the autocorrelation divided by $L$, where $L$ is the length of the binary sequence, so that $R[0] = 1$. The default value is `False`.
        """
        L = self._length
        shifts = np.arange(L) if shifts is None else np.array(shifts)
        autocorrelation = np.array([self._autocorrelation[abs(ell)] if abs(ell) < L else 0 for ell in shifts])
        if normalized:
            return autocorrelation / L
        else:
            return autocorrelation

    def cyclic_autocorrelation(self, shifts=None, normalized=False):
        r"""
        Returns the cyclic autocorrelation $\tilde{R}[\ell]$ of the binary sequence. This is defined as
        $$
            \tilde{R}[\ell] = \sum_{n \in [0:L)} a[n] \tilde{a}_\ell[n],
        $$
        where $a[n]$ is the binary sequence in polar format, and $\tilde{a}\_\ell[n]$ is the sequence $a[n]$ cyclic-shifted by $\ell$ positions. The cyclic autocorrelation $\tilde{R}[\ell]$ is even and periodic with period $L$, where $L$ is the period of the binary sequence.

        Parameters:

            shifts (Optional[Array1D[int]]): A 1D-array containing the values of $\ell$ for which the cyclic autocorrelation will be computed. The default value is `range(L)`, that is, $[0 : L)$.

            normalized (Optional[bool]): If `True`, returns the cyclic autocorrelation divided by $L$, where $L$ is the length of the binary sequence, so that $\tilde{R}[0] = 1$. The default value is `False`.
        """
        L = self._length
        shifts = np.arange(L) if shifts is None else np.array(shifts)
        cyclic_autocorrelation = self._cyclic_autocorrelation[shifts % L]
        if normalized:
            return cyclic_autocorrelation / L
        else:
            return cyclic_autocorrelation
