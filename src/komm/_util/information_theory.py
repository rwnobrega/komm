import math
from functools import partial
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from attrs import field

from .._validation import (
    is_log_base,
    is_pmf,
    is_probability,
    is_transition_matrix,
    validate_call,
)

LogBase = float | Literal["e"]


@validate_call(
    pmf=field(converter=np.asarray, validator=is_pmf),
    base=field(validator=is_log_base),
)
def entropy(
    pmf: npt.ArrayLike,
    base: LogBase = 2.0,
) -> float:
    r"""
    Computes the entropy of a random variable with a given pmf. Let $X$ be a random variable with pmf $p_X$ and alphabet $\mathcal{X}$. Its entropy is given by
    $$
        \mathrm{H}(X) = \sum_{x \in \mathcal{X}} p_X(x) \log \frac{1}{p_X(x)}.
    $$
    By default, the base of the logarithm is $2$, in which case the entropy is measured in bits. For more details, see <cite>CT06, Ch. 2</cite>.

    Parameters:
        pmf (Array1D[float]): The probability mass function $p_X$ of the random variable. It must be a valid pmf, that is, all of its values must be non-negative and sum up to $1$.

        base (Optional[float | str]): The base of the logarithm to be used. It must be a positive float or the string `'e'`. The default value is `2.0`.

    Returns:
        entropy (float): The entropy $\mathrm{H}(X)$ of the random variable.

    Examples:
        >>> komm.entropy([1/4, 1/4, 1/4, 1/4])  # doctest: +NUMBER
        np.float64(2.0)

        >>> komm.entropy(pmf=[1/3, 1/3, 1/3], base=3.0)  # doctest: +NUMBER
        np.float64(1.0)

        >>> komm.entropy([0.5, 0.5], base='e')  # doctest: +NUMBER
        np.float64(0.6931471805599453)
    """
    pmf = cast(npt.NDArray[np.float64], pmf)
    if base == "e":
        return -np.dot(pmf, np.log(pmf, where=(pmf > 0)))
    elif base == 2.0:
        return -np.dot(pmf, np.log2(pmf, where=(pmf > 0)))
    else:
        return -np.dot(pmf, np.log(pmf, where=(pmf > 0))) / np.log(base)


@validate_call(p=field(validator=is_probability))
def binary_entropy(p: float) -> float:
    r"""
    Computes the binary entropy function. For a given probability $p$, it is defined as
    $$
        \Hb(p) = p \log_2 \frac{1}{p} + (1 - p) \log_2 \frac{1}{1 - p},
    $$
    and corresponds to the [entropy](/ref/entropy) of a Bernoulli random variable with parameter $p$.

    Parameters:
        p (float): A probability value. It must satisfy $0 \leq p \leq 1$.

    Returns:
        entropy (float): The value of the binary entropy function $\Hb(p)$.

    Examples:
        >>> [komm.binary_entropy(p) for p in [0.0, 0.25, 0.5, 0.75, 1.0]]  # doctest: +NUMBER
        [0.0, 0.8112781244591328, 1.0, 0.8112781244591328, 0.0]
    """
    if p in {0.0, 1.0}:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


@validate_call(
    input_pmf=field(converter=np.asarray, validator=is_pmf),
    transition_matrix=field(converter=np.asarray, validator=is_transition_matrix),
    base=field(validator=is_log_base),
)
def mutual_information(
    input_pmf: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
    base: LogBase = 2.0,
) -> float:
    output_pmf = np.dot(input_pmf, transition_matrix)
    entropy_output_prior = entropy(output_pmf, base=base)
    entropy_output_posterior = np.dot(
        input_pmf,
        np.apply_along_axis(
            func1d=partial(entropy, base=base),
            axis=1,
            arr=transition_matrix,
        ),
    )
    return entropy_output_prior - entropy_output_posterior


def arimoto_blahut(
    transition_matrix: npt.NDArray[np.float64],
    initial_guess: npt.NDArray[np.float64],
    max_iters: int,
    error_tolerance: float,
) -> npt.NDArray[np.float64]:
    r"""
    Arimoto–Blahut algorithm for channel capacity. See <cite>CT06, Sec. 10.8</cite>.
    """
    p = transition_matrix
    r = initial_guess
    last_r = np.full_like(r, fill_value=np.inf)
    iters = 0
    while iters < max_iters and np.amax(np.abs(r - last_r)) > error_tolerance:
        last_r = r
        q = r[np.newaxis].T * p
        qsum = np.sum(q, axis=0)
        zero_indices = qsum == 0
        qsum_copy = qsum.copy()
        qsum_copy[zero_indices] = 1
        q /= qsum_copy
        q[:, zero_indices] = 0
        r = np.prod(q**p, axis=1)
        r /= np.sum(r, axis=0)
        iters += 1
    return r
