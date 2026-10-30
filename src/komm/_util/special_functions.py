from collections.abc import Iterable
from math import erfc, exp, floor, inf, isinf, lgamma, log, log1p, nan, sqrt
from operator import index
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from .validators import validate_float_array, validate_positive_integer

norm = NormalDist()


def _gaussian_q(x: float) -> float:
    # Equivalent to `norm.cdf(-x)``, but accurate in the tail for all supported Python versions.
    # `NormalDist.cdf`` only became erfc-based (and thus tail-accurate) in Python 3.14.
    # See https://github.com/python/cpython/issues/132893
    return 0.5 * erfc(x / sqrt(2))


def gaussian_q(x: npt.ArrayLike) -> npt.NDArray[np.floating] | float:
    r"""
    Computes the Gaussian Q-function. It is given by
    $$
        \mathrm{Q}(x) = \frac{1}{\sqrt{2\pi}} \int_x^\infty \mathrm{e}^{-u^2/2} \, \mathrm{d}u.
    $$
    This corresponds to the complementary cdf of the standard Gaussian distribution. For more details, see [Wikipedia: Q-function](https://en.wikipedia.org/wiki/Q-function).

    Parameters:
        x: The input to the function. Should be a float or array of floats.

    Returns:
        y: The value $y = \mathrm{Q}(x)$.

    Examples:
        >>> komm.gaussian_q(0.0)
        0.5

        >>> komm.gaussian_q([[-1.0], [0.0], [1.0]])
        array([[0.84134475],
               [0.5       ],
               [0.15865525]])
    """
    result = np.vectorize(_gaussian_q)(x)
    return float(result) if np.ndim(result) == 0 else result


def _gaussian_q_inv(y: float) -> float:
    if y == 0:
        return np.inf
    if y == 1:
        return -np.inf
    return -norm.inv_cdf(y)


def gaussian_q_inv(y: npt.ArrayLike) -> npt.NDArray[np.floating] | float:
    r"""
    Computes the inverse of the [Gaussian Q-function](/ref/gaussian_q). More precisely, it computes the value of $x$ such that $\mathrm{Q}(x) = y$.

    Parameters:
        y: The input to the function. Should be a float or array of floats in the real interval $[0, 1]$.

    Returns:
        x: The value $x = \mathrm{Q}^{-1}(y)$.

    Examples:
        >>> komm.gaussian_q_inv(0.5)
        0.0

        >>> komm.gaussian_q_inv([[0.841344746], [0.5], [0.158655254]])
        array([[-1.],
               [ 0.],
               [ 1.]])
    """
    y = np.asarray(y)
    if not np.all((0 <= y) & (y <= 1)):
        raise ValueError("'y' must be in [0, 1]")
    result = np.vectorize(_gaussian_q_inv)(y) + 0.0  # + 0.0 to avoid -0.0
    return float(result) if np.ndim(result) == 0 else result


def _poisson_log_pmf(k: int, mu: float) -> float:
    return k * log(mu) - mu - lgamma(k + 1) if k > 0 else -mu


def _skellam_log_cdf(mu: float, nu: float, s: int, tol: float = 1e-16) -> float:
    # log P(N_nu - N_mu < s), with Poisson N_mu and N_nu.
    # Sums P(N_mu = j) P(N_nu < s + j) over j.
    j = max(0, 1 - s, floor(mu - 9 * sqrt(mu)))  # Skip negligible weights
    n = s + j
    head = c = 1.0  # P(N_nu < n) / P(N_nu = n - 1)
    for k in range(n - 1, 0, -1):
        c *= k / nu
        head += c
        q = (k - 1) / nu
        if c * q < tol * (1 - q) * head:
            break
    log_peak = _poisson_log_pmf(j, mu) + _poisson_log_pmf(n - 1, nu) + log(head)
    rho = nu / (n * head)  # P(N_nu = n) / P(N_nu < n)
    term = total = 1.0  # Relative to largest term so far
    while True:
        r = mu / (j + 1) * (1 + rho)  # Next term over current term
        if term * r < tol * (1 - r) * total:  # Log-concave terms: tail bound
            break
        term *= r
        if term > 1:
            log_peak += log(term)
            total /= term
            term = 1.0
        total += term
        rho *= nu / (n + 1) / (1 + rho)
        j, n = j + 1, n + 1
    return log_peak + log(total)


def _marcum_q(m: int, a: float, x: float) -> float:
    if isinf(a) or isinf(x):
        return 0.0 if a < x else 1.0 if a > x else nan
    # Q = P(N_beta - N_lambda < m), with Poisson N's
    lam, beta = a**2 / 2, x**2 / 2
    if beta == 0:
        return 1.0
    if beta > lam + m:  # Q is the smaller tail
        return exp(_skellam_log_cdf(lam, beta, m))
    return 1 - exp(_skellam_log_cdf(beta, lam, 1 - m))


def marcum_q(
    m: int, a: npt.ArrayLike, x: npt.ArrayLike
) -> npt.NDArray[np.floating] | float:
    r"""
    Computes the Marcum Q-function. It is given by
    $$
        \mathrm{Q}_{m,a}(x) = \frac{1}{a^{m-1}} \int_x^\infty u^m \exp \left( -\frac{u^2 + a^2}{2} \right) I\_{m-1}(a u) \, \mathrm{d}u,
    $$
    where $I$ is the modified Bessel function of the first kind. This corresponds to the complementary cdf of the non-central chi distribution with $2m$ degrees of freedom and non-centrality parameter $a$. For more details, see [Wikipedia: Marcum Q-function](https://en.wikipedia.org/wiki/Marcum_Q-function).

    Parameters:
        m: The order of the Marcum Q-function. Should be a positive integer.
        a: The value of $a$. Should be a float or array of floats. Must satisfy $a \geq 0$.
        x: The input to the function. Should be a float or array of floats. Must satisfy $x \geq 0$.

    Returns:
        y: The value $y = \mathrm{Q}_{m,a}(x)$.

    Examples:
        >>> komm.marcum_q(1, 1, 1)
        0.7328798037968203

        >>> komm.marcum_q(2, 0.5, [1.2, 1.4, 1.6])
        array([0.85225816, 0.76472056, 0.66139663])
    """
    m = validate_positive_integer(m)
    a = validate_float_array(a)
    if not np.all(a >= 0):
        raise ValueError("'a' must be non-negative")
    x = validate_float_array(x)
    if not np.all(x >= 0):
        raise ValueError("'x' must be non-negative")
    result = np.vectorize(_marcum_q)(m, a, x)
    return float(result) if np.ndim(result) == 0 else result


def _marcum_q_inv(m: int, a: float, y: float) -> float:
    if y == 1:
        return 0.0
    if y == 0 or isinf(a):
        return inf
    lam = a**2 / 2
    upper = y < 0.5  # Solve for the smaller tail

    def residual(v: float) -> float:  # Decreasing in v = log(x)
        beta = exp(2 * v) / 2
        if beta > lam + m:
            log_q = _skellam_log_cdf(lam, beta, m)
            log_p = log1p(-exp(log_q))
        else:
            log_p = _skellam_log_cdf(beta, lam, 1 - m)
            log_q = log1p(-exp(log_p))
        return log_q - log(y) if upper else log1p(-y) - log_p

    # Bracket: R = |a + Z| is 1-Lipschitz in Z
    lo = max(
        sqrt(a**2 + 2 * m - 1) - sqrt(-2 * log1p(-y)),  # Var(R) <= 1
        sqrt(2) * exp((lgamma(m + 1) + log1p(-y)) / (2 * m)),  # P <= beta^m / m!
    )
    hi = sqrt(a**2 + 2 * m) + sqrt(-2 * log(y))
    v_lo, v_hi = log(lo), log(hi)
    r_lo, r_hi = residual(v_lo), residual(v_hi)
    v, side = v_lo, 0
    for _ in range(100):  # Illinois method
        v = (v_lo * r_hi - v_hi * r_lo) / (r_hi - r_lo)
        if not v_lo < v < v_hi:  # Float precision reached
            break
        r = residual(v)
        if r > 0:
            v_lo, r_lo = v, r
            if side > 0:
                r_hi /= 2
            side = 1
        else:
            v_hi, r_hi = v, r
            if side < 0:
                r_lo /= 2
            side = -1
    return exp(v)


def marcum_q_inv(
    m: int, a: npt.ArrayLike, y: npt.ArrayLike
) -> npt.NDArray[np.floating] | float:
    r"""
    Computes the inverse of the [Marcum Q-function](/ref/marcum_q) with respect to $x$. More precisely, it computes the value of $x \geq 0$ such that $\mathrm{Q}_{m,a}(x) = y$.

    Parameters:
        m: The order of the Marcum Q-function. Should be a positive integer.
        a: The value of $a$. Should be a float or array of floats. Must satisfy $a \geq 0$.
        y: The input to the function. Should be a float or array of floats in the real interval $[0, 1]$.

    Returns:
        x: The value $x = \mathrm{Q}_{m,a}^{-1}(y)$.

    Examples:
        >>> komm.marcum_q_inv(1, 1, 0.7328798037968203)  # doctest: +FLOAT_CMP
        1.0

        >>> komm.marcum_q_inv(2, 0.5, [0.852258163, 0.764720559, 0.66139663])
        array([1.2, 1.4, 1.6])
    """
    m = validate_positive_integer(m)
    a = validate_float_array(a)
    if not np.all(a >= 0):
        raise ValueError("'a' must be non-negative")
    y = validate_float_array(y)
    if not np.all((0 <= y) & (y <= 1)):
        raise ValueError("'y' must be in [0, 1]")
    result = np.vectorize(_marcum_q_inv)(m, a, y)
    return float(result) if np.ndim(result) == 0 else result


def logcosh(x: npt.ArrayLike) -> np.floating:
    # https://stackoverflow.com/a/57786270/
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)


def boxplus(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.floating] | float:
    r"""
    Computes the box-plus operation. It is defined by
    $$
        a \boxplus b = 2 \operatorname{atanh} \left( \tanh\left(\frac{a}{2}\right) \tanh\left(\frac{b}{2}\right) \right).
    $$
    If $X, Y$ are independent binary random variables, then $L(X \oplus Y) = L(X) \boxplus L(Y)$, where $L$ denotes the L-value and $\oplus$ denotes the modulo-$2$ sum.

    Parameters:
        a: A float or array of floats containing L-values.
        b: A float or array of floats containing L-values.

    Returns:
        The result of the boxplus operation. It has the same shape as the inputs.

    Examples:
        >>> komm.boxplus(1, 2)  # doctest: +FLOAT_CMP
        0.7353256640555192

        >>> komm.boxplus([0, 1, 2], [1, -1, -1])
        array([ 0.        , -0.43378083, -0.73532566])
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    s = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
    p1 = np.exp(-np.abs(a + b))
    p2 = np.exp(-np.abs(a - b))
    result = s + np.log1p(p1) - np.log1p(p2)
    return float(result) if np.ndim(result) == 0 else result


def macwilliams_transform(distribution: Iterable[int]) -> list[int]:
    r"""
    Computes the MacWilliams transform of a weight distribution. Let $A_0, A_1, \ldots, A_n$ be the codeword weight distribution of a binary linear code with dimension $k$, in which $A_w$ is the number of codewords of Hamming weight $w$. The codeword weight distribution $B_0, B_1, \ldots, B_n$ of the dual code is given by the MacWilliams identity
    $$
        \sum_{w=0}^n B_w z^w = \frac{1}{2^k} \sum_{w=0}^n A_w (1 - z)^w (1 + z)^{n - w}.
    $$
    For more details, see <cite>LC04, Sec. 3.6</cite>.

    Parameters:
        distribution: The codeword weight distribution $A_0, A_1, \ldots, A_n$ of the code. Must be a list of non-negative integers.

    Returns:
        The codeword weight distribution $B_0, B_1, \ldots, B_n$ of the dual code.

    Examples:
        >>> komm.macwilliams_transform([1, 0, 0, 7, 7, 0, 0, 1])
        [1, 0, 0, 0, 7, 0, 0, 0]
    """
    distribution = [index(a) for a in distribution]
    n, size = len(distribution) - 1, sum(distribution)
    if not (size > 0 and min(distribution) >= 0):
        raise ValueError("'distribution' must be non-negative and nonzero")
    dual = [0] * (n + 1)
    for i, a in enumerate(distribution):
        if a == 0:
            continue
        # Coefficients of (1 - z)^i (1 + z)^(n - i).
        prev, curr = 0, 1
        for j in range(n + 1):
            dual[j] += a * curr
            prev, curr = curr, ((n - 2 * i) * curr - (n - j + 1) * prev) // (j + 1)
    if not all(b >= 0 and b % size == 0 for b in dual):
        raise ValueError("'distribution' must come from a linear code")
    return [b // size for b in dual]
