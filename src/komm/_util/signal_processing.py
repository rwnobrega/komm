from typing import Any

import numpy as np
import numpy.typing as npt


def sampling_rate_compress(
    input: npt.ArrayLike,
    factor: int,
    offset: int = 0,
    axis: int = -1,
) -> npt.NDArray[Any]:
    r"""
    Performs sampling rate compression (downsampling). For a given input $x[n]$, the output is
    $$
    y[n] = x[n M + \Delta],
    $$
    where $M$ is the *compression factor* and $\Delta \in [0:M)$ is the *offset* for the first selected element. In words, the compressor extracts every $M$-th element starting from the offset $\Delta$ along the specified axis. For more details, see <cite>OS99, Sec. 4.6.1</cite>.

    Parameters:
        input: The input array $x[n]$ to be compressed.
        factor: The compression factor $M$.
        offset: The offset $\Delta$. Must satisfy $\Delta \in [0:M)$.
        axis: The axis along which to extract elements. Default is the last axis.

    Returns:
        output: The compressed array $y[n]$.

    Examples:
        >>> komm.sampling_rate_compress(
        ...    [[11, 12, 13, 14, 15],
        ...     [16, 17, 18, 19, 20]],
        ...    factor=3,
        ... )
        array([[11, 14],
               [16, 19]])
        >>> komm.sampling_rate_compress(
        ...    [[11, 12, 13, 14, 15],
        ...     [16, 17, 18, 19, 20]],
        ...    factor=3,
        ...    offset=1,
        ... )
        array([[12, 15],
               [17, 20]])
        >>> komm.sampling_rate_compress(
        ...     [[11, 12],
        ...      [13, 14],
        ...      [15, 16],
        ...      [17, 18],
        ...      [19, 20]],
        ...     factor=3,
        ...     axis=0,
        ... )
        array([[11, 12],
               [17, 18]])
    """
    if not factor > 0:
        raise ValueError("'factor' should be a positive integer")
    if not (0 <= offset < factor):
        raise ValueError("'offset' should satisfy 0 <= offset < factor")
    input = np.asarray(input)
    indexer = [slice(None)] * input.ndim
    indexer[axis] = slice(offset, input.shape[axis], factor)
    return input[tuple(indexer)]


def sampling_rate_expand(
    input: npt.ArrayLike,
    factor: int,
    offset: int = 0,
    axis: int = -1,
) -> npt.NDArray[Any]:
    r"""
    Performs sampling rate expansion (upsampling). For a given input $x[n]$, the output is
    $$
    y[n] = \begin{cases}
    x[n \operatorname{div} L] & \text{if } n \bmod L = \Delta, \\\\
    0,       & \text{otherwise}
    \end{cases}
    $$
    where $L$ is the *expansion factor* and $\Delta \in [0:L)$ is the *offset* for the first output element. In words, the expander inserts $L-1$ zeros between each element of the input array along the specified axis, starting from the offset $\Delta$. For more details, see <cite>OS99, Sec. 4.6.2</cite>.

    Parameters:
        input: The input array $x[n]$ to be expanded.
        factor: The expansion factor $L$.
        offset: The offset $\Delta$. Must satisfy $\Delta \in [0:L)$.
        axis: The axis along which to insert zeros. Default is the last axis.

    Returns:
        output: The expanded array $y[n]$.

    Examples:
        >>> komm.sampling_rate_expand([[1, 2], [3, 4]], factor=3)
        array([[1, 0, 0, 2, 0, 0],
               [3, 0, 0, 4, 0, 0]])
        >>> komm.sampling_rate_expand([[1, 2], [3, 4]], factor=3, offset=1)
        array([[0, 1, 0, 0, 2, 0],
               [0, 3, 0, 0, 4, 0]])
        >>> komm.sampling_rate_expand([[1, 2], [3, 4]], factor=3, axis=0)
        array([[1, 2],
               [0, 0],
               [0, 0],
               [3, 4],
               [0, 0],
               [0, 0]])
    """
    if not factor > 0:
        raise ValueError("'factor' should be a positive integer")
    if not (0 <= offset < factor):
        raise ValueError("'offset' should satisfy 0 <= offset < factor")
    input = np.asarray(input)
    shape = list(input.shape)
    shape[axis] *= factor
    output = np.zeros(shape, dtype=input.dtype)
    indexer = [slice(None)] * input.ndim
    indexer[axis] = slice(offset, shape[axis], factor)
    output[tuple(indexer)] = input
    return output
