import math
from functools import partial
from typing import Literal

import numpy as np
import numpy.typing as npt

LogBase = float | Literal["e"]


def assert_is_log_base(value: float | str) -> None:
    if (isinstance(value, str) and value != "e") or (
        isinstance(value, float) and value <= 0.0
    ):
        raise ValueError("log base must be 'e' or a positive float")


def assert_is_probability(value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError("probability must be between 0 and 1")


class PMF(npt.NDArray[np.floating]):
    def __new__(cls, values: npt.ArrayLike):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError("PMF must be a 1D array")
        if not np.all(arr >= 0.0):
            raise ValueError("PMF must be non-negative")
        if not np.isclose(arr.sum(), 1.0):
            raise ValueError("PMF must sum to 1.0")
        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj: npt.NDArray[np.floating] | None) -> None:
        if obj is None:
            return


def random_pmf(size: int) -> PMF:
    pmf = np.random.rand(size)
    return PMF(pmf / pmf.sum())


class TransitionMatrix(npt.NDArray[np.floating]):
    def __new__(cls, values: npt.ArrayLike):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Transition matrix must be a 2D array")
        if not np.all(arr >= 0.0):
            raise ValueError("Transition matrix must be non-negative")
        if not np.allclose(arr.sum(axis=1), 1.0):
            raise ValueError("Rows of transition matrix must sum to 1.0")
        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj: npt.NDArray[np.floating] | None) -> None:
        if obj is None:
            return


def entropy(pmf: npt.ArrayLike, base: LogBase = 2.0) -> float:
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
    pmf = PMF(pmf)
    assert_is_log_base(base)
    if base == "e":
        return -np.dot(pmf, np.log(pmf, where=(pmf > 0)))
    elif base == 2.0:
        return -np.dot(pmf, np.log2(pmf, where=(pmf > 0)))
    else:
        return -np.dot(pmf, np.log(pmf, where=(pmf > 0))) / np.log(base)


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
    assert_is_probability(p)
    if p in {0.0, 1.0}:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def relative_entropy(
    pmf: npt.ArrayLike,
    qmf: npt.ArrayLike,
    base: LogBase = 2.0,
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
    pmf = PMF(pmf)
    qmf = PMF(qmf)
    assert_is_log_base(base)
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
    base: LogBase = 2.0,
) -> float:
    input_pmf = PMF(input_pmf)
    assert_is_log_base(base)
    transition_matrix = TransitionMatrix(transition_matrix)
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


def marginalize_bits(
    symbol_metrics: npt.NDArray[np.floating],
    labeling: npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:
    r"""
    Marginalizes symbol metrics to $L$-values of bits.
    """
    mask0 = labeling == 0
    mask1 = labeling == 1
    p0 = mask0.T @ symbol_metrics
    p1 = mask1.T @ symbol_metrics
    with np.errstate(divide="ignore"):
        return np.log(p0) - np.log(p1)
