import functools

import numpy as np
import numpy.typing as npt


class BinarySequence:
    r"""
    General binary sequence. It may be represented either in *bit format*, denoted by :math:`b[n]`, with elements in the set :math:`\{ 0, 1 \}`, or in *polar format*, denoted by :math:`a[n]`, with elements in the set :math:`\{ \pm 1 \}`. The correspondences :math:`0 \mapsto +1` and :math:`1 \mapsto -1` from bit format to polar format is assumed.
    """

    def __init__(self, **kwargs):
        r"""
        Constructor for the class. It expects *exactly one* the following parameters:

        Parameters:

            bit_sequence (1D-array of :obj:`int`): The binary sequence in bit format. Must be an 1D-array with elements in :math:`\{ 0, 1 \}`.

            polar_sequence (1D-array of :obj:`int`): The binary sequence in polar format. Must be an 1D-array with elements in :math:`\{ \pm 1 \}`.
        """
        kwargs_set = set(kwargs.keys())
        if kwargs_set == {"bit_sequence"}:
            self._bit_sequence = np.array(kwargs["bit_sequence"], dtype=int)
            self._polar_sequence = (-1) ** self._bit_sequence
            self._constructed_from = "bit_sequence"
        elif kwargs_set == {"polar_sequence"}:
            self._polar_sequence = np.array(kwargs["polar_sequence"], dtype=int)
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
        The binary sequence in bit format, :math:`b[n] \in \{ 0, 1 \}`.
        """
        return self._bit_sequence

    @property
    def polar_sequence(self):
        r"""
        The binary sequence in polar format, :math:`a[n] \in \{ \pm 1 \}`.
        """
        return self._polar_sequence

    @property
    def length(self):
        r"""
        The length (or period) :math:`L` of the binary sequence.
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
        Returns the autocorrelation :math:`R[\ell]` of the binary sequence. This is defined as

        .. math::
           R[\ell] = \sum_{n \in \mathbb{Z}} a[n] a_{\ell}[n],

        where :math:`a[n]` is the binary sequence in polar format, and :math:`a_{\ell}[n] = a[n - \ell]` is the sequence :math:`a[n]` shifted by :math:`\ell` positions. The autocorrelation :math:`R[\ell]` is even and satisfies :math:`R[\ell] = 0` for :math:`|\ell| \geq L`, where :math:`L` is the length of the binary sequence.

        Parameters:

            shifts (1D-array of :obj:`int`, optional): An 1D array containing the values of :math:`\ell` for which the autocorrelation will be computed. The default value is :code:`range(L)`, that is, :math:`[0 : L)`.

            normalized: (:obj:`boolean`, optional): If :code:`True`, returns the autocorrelation divided by :math:`L`, where :math:`L` is the length of the binary sequence, so that :math:`R[0] = 1`. The default value is :code:`False`.
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
        Returns the cyclic autocorrelation :math:`\tilde{R}[\ell]` of the binary sequence. This is defined as

        .. math::
           \tilde{R}[\ell] = \sum_{n \in [0:L)} a[n] \tilde{a}_{\ell}[n],

        where :math:`a[n]` is the binary sequence in polar format, and :math:`\tilde{a}_{\ell}[n]` is the sequence :math:`a[n]` cyclic-shifted by :math:`\ell` positions. The cyclic autocorrelation :math:`\tilde{R}[\ell]` is even and periodic with period :math:`L`, where :math:`L` is the period of the binary sequence.

        Parameters:

            shifts (1D-array of :obj:`int`, optional): An 1D array containing the values of :math:`\ell` for which the cyclic autocorrelation will be computed. The default value is :code:`range(L)`, that is, :math:`[0 : L)`.

            normalized (:obj:`boolean`, optional): If :code:`True`, returns the cyclic autocorrelation divided by :math:`L`, where :math:`L` is the length of the binary sequence, so that :math:`\tilde{R}[0] = 1`. The default value is :code:`False`.
        """
        L = self._length
        shifts = np.arange(L) if shifts is None else np.array(shifts)
        cyclic_autocorrelation = self._cyclic_autocorrelation[shifts % L]
        if normalized:
            return cyclic_autocorrelation / L
        else:
            return cyclic_autocorrelation
