from typing import Iterable, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=Union[np.float64, np.complex128])


def acorr(
    seq: npt.ArrayLike,
    shifts: Optional[Iterable[int]] = None,
    normalized: bool = False,
) -> npt.NDArray[DType]:
    r"""
    Computes the autocorrelation $R[\ell]$ of a real or complex sequence $x[n]$. This is defined as
    $$
        R[\ell] = \sum_{n \in \mathbb{Z}} x[n] x^*_\ell[n],
    $$
    where $x^\*\_\ell[n] = x^\*[n - \ell]$ is the complex conjugate of $x[n]$ shifted by $\ell$ positions. The autocorrelation $R[\ell]$ is even symmetric and satisfies $R[\ell] = 0$ for $|\ell| \geq L$, where $L$ is the length of the sequence.

    Parameters:
        seq (Array1D[float | complex]): A 1D-array containing the sequence $x[n]$, of length $L$.

        shifts (Optional[Array1D[int]]): A 1D-array containing the values of $\ell$ for which the autocorrelation will be computed. The default value is `range(len(seq))`, that is, $[0 : L)$.

        normalized (Optional[bool]): If `True`, returns the autocorrelation divided by the sequence energy, so that $R[0] = 1$. The default value is `False`.

    Returns:
        acorr (SameAsInput): The autocorrelation $R[\ell]$ of the sequence.

    Examples:
        >>> komm.acorr([1.0, 2.0, 3.0, 4.0], shifts=[-2, -1, 0, 1, 2])
        array([11., 20., 30., 20., 11.])
    """
    seq = np.asarray(seq)
    seq_conj = np.conj(seq)
    shifts = np.arange(seq.size) if shifts is None else shifts
    acorr = np.empty_like(shifts, dtype=seq.dtype)
    for i, ell in enumerate(shifts):
        if ell < 0:
            acorr[i] = np.dot(seq[:ell], seq_conj[-ell:])
        elif ell > 0:
            acorr[i] = np.dot(seq[ell:], seq_conj[:-ell])
        else:
            acorr[i] = np.dot(seq, seq_conj)
    energy = np.dot(seq, seq_conj)
    return acorr / energy if normalized else acorr


def cyclic_acorr(
    seq: npt.ArrayLike,
    shifts: Optional[Iterable[int]] = None,
    normalized: bool = False,
) -> npt.NDArray[DType]:
    r"""
    Computes the cyclic autocorrelation $\tilde{R}[\ell]$ of a real or complex sequence $x[n]$. This is defined as
    $$
        \tilde{R}[\ell] = \sum_{n \in [0:L)} x[n] \tilde{x}^*_\ell[n],
    $$
    where $\tilde{x}^\*\_\ell[n]$ is the complex conjugate of $x[n]$ cyclic-shifted by $\ell$ positions, and $L$ is the period of the sequence. The cyclic autocorrelation $\tilde{R}[\ell]$ is even symmetric and periodic with period $L$.

    Parameters:
        seq (Array1D[float | complex]): A 1D-array containing the sequence $x[n]$, of length $L$.

        shifts (Optional[Array1D[int]]): A 1D-array containing the values of $\ell$ for which the cyclic autocorrelation will be computed. The default value is `range(len(seq))`, that is, $[0 : L)$.

        normalized (Optional[bool]): If `True`, returns the cyclic autocorrelation divided by the sequence energy, so that $R[0] = 1$. The default value is `False`.

    Returns:
        cyclic_acorr (SameAsInput): The cyclic autocorrelation $\tilde{R}[\ell]$ of the sequence.

    Examples:
        >>> komm.cyclic_acorr([1.0, 2.0, 3.0, 4.0], shifts=[-2, -1, 0, 1, 2])
        array([22., 24., 30., 24., 22.])
    """
    seq = np.asarray(seq)
    shifts = np.arange(seq.size) if shifts is None else shifts
    conj_seq = np.conj(seq)
    acorr = np.array([np.dot(seq, np.roll(seq, ell).conj()) for ell in shifts])
    energy = np.dot(seq, conj_seq)
    return acorr / energy if normalized else acorr
