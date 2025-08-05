from functools import partial
from math import log2
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..types import Array1D
from .validators import (
    validate_log_base,
    validate_pmf,
    validate_probability,
    validate_transition_matrix,
)


def random_pmf(size: int) -> Array1D[np.floating]:
    pmf = np.random.rand(size)
    return pmf / pmf.sum()


def entropy(pmf: npt.ArrayLike, base: float | Literal["e"] = 2.0) -> float:
    r"""
    Computes the entropy of a random variable with a given pmf. Let $X$ be a random variable with pmf $p_X$ and alphabet $\mathcal{X}$. Its entropy is given by
    $$
        \mathrm{H}(X) = \sum_{x \in \mathcal{X}} p_X(x) \log \frac{1}{p_X(x)}.
    $$
    By default, the base of the logarithm is $2$, in which case the entropy is measured in bits. For more details, see <cite>CT06, Sec. 2.1</cite>.

    Parameters:
        pmf: The probability mass function $p_X$ of the random variable. It must be a valid pmf, that is, all of its values must be non-negative and sum up to $1$.

        base: The base of the logarithm to be used. It must be a positive float or the string `'e'`. The default value is `2.0`.

    Returns:
        The entropy $\mathrm{H}(X)$ of the random variable.

    Examples:
        >>> komm.entropy([1/4, 1/4, 1/4, 1/4])  # doctest: +FLOAT_CMP
        np.float64(2.0)

        >>> komm.entropy(pmf=[1/3, 1/3, 1/3], base=3.0)  # doctest: +FLOAT_CMP
        np.float64(1.0)

        >>> komm.entropy([0.5, 0.5], base='e')  # doctest: +FLOAT_CMP
        np.float64(0.6931471805599453)
    """
    pmf = validate_pmf(pmf)
    validate_log_base(base)
    pmf1 = pmf[pmf > 0]
    if base == "e":
        log_pmf1 = np.log(pmf1)
    elif base == 2.0:
        log_pmf1 = np.log2(pmf1)
    else:
        log_pmf1 = np.log(pmf1) / np.log(base)
    return -np.dot(pmf1, log_pmf1)


def binary_entropy(p: float) -> float:
    r"""
    Computes the binary entropy function. For a given probability $p$, it is defined as
    $$
        \Hb(p) = p \log_2 \frac{1}{p} + (1 - p) \log_2 \frac{1}{1 - p},
    $$
    and corresponds to the [entropy](/ref/entropy) of a Bernoulli random variable with parameter $p$.

    Parameters:
        p: A probability value. It must satisfy $0 \leq p \leq 1$.

    Returns:
        The value of the binary entropy function $\Hb(p)$.

    Examples:
        >>> [komm.binary_entropy(p) for p in [0.0, 0.25, 0.5, 0.75, 1.0]]  # doctest: +FLOAT_CMP
        [0.0, 0.8112781244591328, 1.0, 0.8112781244591328, 0.0]
    """
    validate_probability(p)
    if p in {0.0, 1.0}:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def binary_entropy_inv(h: float, tol: float = 1e-12) -> float:
    r"""
    Computes the inverse of the [binary entropy function](/ref/binary_entropy). More precisely, it computes the value of $p \in [0, 1/2]$ such that $\Hb(p) = h$.

    Parameters:
        h: A value in the interval $[0, 1]$.
        tol: The tolerance for the binary search. The default value is `1e-12`.

    Returns:
        The value of $p \in [0, 1/2]$ such that $\Hb(p) = h$.

    Examples:
        >>> [komm.binary_entropy_inv(h) for h in [0.0, 0.25, 0.5, 0.75, 1.0]]
        [0.0, 0.04169269027397604, 0.1100278644385071, 0.2145017448597173, 0.5]
    """
    if not 0.0 <= h <= 1.0:
        raise ValueError("h must be in [0, 1]")
    if h in {0.0, 1.0}:
        return 0.5 * h
    low, high = 0.0, 0.5
    while high - low > tol:
        mid = (low + high) / 2
        if binary_entropy(mid) < h:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def relative_entropy(
    pmf: npt.ArrayLike,
    qmf: npt.ArrayLike,
    base: float | Literal["e"] = 2.0,
) -> float:
    r"""
    Computes the relative entropy (Kullback–Leibler divergence) between two <span>pmf</span>s. Let $p$ and $q$ be two <span>pmf</span>s over the same alphabet $\mathcal{X}$. The relative entropy of $p$ with respect to $q$ is defined as
    $$
        \mathrm{D}(p \parallel q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}.
    $$
    Note that, in general, $\mathrm{D}(p \parallel q) \neq \mathrm{D}(q \parallel p)$. For more details, see <cite>CT06, Sec. 2.3</cite>.

    Parameters:
        pmf: The probability mass function $p$. It must be a valid pmf, that is, all of its values must be non-negative and sum up to $1$.

        qmf: The probability mass function $q$. It must be a valid pmf, that is, all of its values must be non-negative and sum up to $1$.

        base: The base of the logarithm to be used. It must be a positive float or the string `'e'`. The default value is `2.0`.

    Returns:
        The relative entropy $\mathrm{D}(p \parallel q)$ between the two <span>pmf</span>s.

    Examples:
        >>> komm.relative_entropy([1/2, 1/2], [1/2, 1/2])  # doctest: +FLOAT_CMP
        np.float64(0.0)

        >>> komm.relative_entropy([1/2, 1/2], [3/4, 1/4])  # doctest: +FLOAT_CMP
        np.float64(0.20751874963942185)

        >>> komm.relative_entropy([3/4, 1/4], [1/2, 1/2])  # doctest: +FLOAT_CMP
        np.float64(0.18872187554086717)

        >>> komm.relative_entropy([1/2, 1/2], [0, 1])  # doctest: +FLOAT_CMP
        np.float64(inf)
    """
    pmf = validate_pmf(pmf)
    qmf = validate_pmf(qmf)
    validate_log_base(base)
    with np.errstate(divide="ignore"):
        if base == "e":
            return np.dot(pmf, np.log(pmf / qmf))
        elif base == 2.0:
            return np.dot(pmf, np.log2(pmf / qmf))
        else:
            return np.dot(pmf, np.log(pmf / qmf)) / np.log(base)


def mutual_information(
    input_pmf: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
    base: float | Literal["e"] = 2.0,
) -> float:
    input_pmf = validate_pmf(input_pmf)
    validate_log_base(base)
    transition_matrix = validate_transition_matrix(transition_matrix)
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
    transition_matrix: npt.NDArray[np.floating],
    initial_guess: npt.NDArray[np.floating],
    max_iter: int,
    tol: float,
) -> npt.NDArray[np.floating]:
    r"""
    Arimoto–Blahut algorithm for channel capacity. See <cite>CT06, Sec. 10.8</cite>.
    """
    p = transition_matrix
    r = initial_guess
    last_r = np.full_like(r, fill_value=np.inf)
    iters = 0
    while iters < max_iter and np.amax(np.abs(r - last_r)) > tol:
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
