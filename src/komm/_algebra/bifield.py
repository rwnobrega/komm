from functools import cache

import numpy as np
import numpy.typing as npt

from .._util.bit_operations import int_to_bits
from .FiniteBifield import FiniteBifield


def multiply_mod(x: int, y: int, modulus: int) -> int:
    # Field product on integer representations.
    order = 1 << (modulus.bit_length() - 1)
    result = 0
    while y:
        if y & 1:
            result ^= x
        y >>= 1
        x <<= 1
        if x >= order:
            x ^= modulus
    return result


@cache
def exp_table(field: FiniteBifield) -> npt.NDArray[np.integer]:
    # α^i for i in [0 : 2n), avoiding mod n of indices.
    n = field.order - 1
    modulus, alpha = int(field.modulus), int(field.primitive_element)
    table = np.empty(2 * n, dtype=int)
    x = 1
    for i in range(n):
        table[i] = x
        x = multiply_mod(x, alpha, modulus)
    table[n:] = table[:n]
    return table


@cache
def log_table(field: FiniteBifield) -> npt.NDArray[np.integer]:
    # log_α(x) for x in [1 : n], dummy at 0.
    n = field.order - 1
    table = np.zeros(field.order, dtype=int)
    table[exp_table(field)[:n]] = np.arange(n)
    return table


def multiply(
    field: FiniteBifield, x: npt.ArrayLike, y: npt.ArrayLike
) -> npt.NDArray[np.integer]:
    r"""
    Multiplies elements of a finite field. The elements are given by their integer representations, in $[0 : 2^k)$. The operation is elementwise, with broadcasting.

    Parameters:
        field: A finite field.
        x: The first factor.
        y: The second factor.

    Returns:
        product: The product $x y$.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> multiply(field, [0b1011, 0b0110], 0b1100)
        array([13, 14])
    """
    x, y = np.asarray(x), np.asarray(y)
    exp, log = exp_table(field), log_table(field)
    return np.where((x == 0) | (y == 0), 0, exp[log[x] + log[y]])


def divide(
    field: FiniteBifield, x: npt.ArrayLike, y: npt.ArrayLike
) -> npt.NDArray[np.integer]:
    r"""
    Divides elements of a finite field. The elements are given by their integer representations, in $[0 : 2^k)$. The operation is elementwise, with broadcasting.

    Parameters:
        field: A finite field.
        x: The dividend.
        y: The divisor.

    Returns:
        quotient: The quotient $x / y$.

    Raises:
        ZeroDivisionError: If the divisor is zero.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> divide(field, [0b1011, 0b0110], 0b1100)
        array([2, 9])
    """
    x, y = np.asarray(x), np.asarray(y)
    if np.any(y == 0):
        raise ZeroDivisionError("division by zero")
    n = field.order - 1
    exp, log = exp_table(field), log_table(field)
    return np.where(x == 0, 0, exp[log[x] - log[y] + n])


def power(
    field: FiniteBifield, b: npt.ArrayLike, e: npt.ArrayLike
) -> npt.NDArray[np.integer]:
    r"""
    Raises elements of a finite field to integer powers. The elements are given by their integer representations, in $[0 : 2^k)$. The operation is elementwise, with broadcasting.

    Parameters:
        field: A finite field.
        b: The base.
        e: The exponent.

    Returns:
        power: The power $b^e$.

    Raises:
        ZeroDivisionError: If the base is zero and the exponent is negative.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> power(field, 0b10, [0, 1, 2, 3, 4, 15, -1])
        array([1, 2, 4, 8, 3, 1, 9])
    """
    b, e = np.asarray(b), np.asarray(e)
    if np.any((b == 0) & (e < 0)):
        raise ZeroDivisionError("zero cannot be raised to a negative power")
    n = field.order - 1
    exp, log = exp_table(field), log_table(field)
    result = exp[log[b] * (e % n) % n]
    return np.where(b == 0, np.where(e == 0, 1, 0), result)


# Polynomial functions


def horner(
    field: FiniteBifield,
    coefficients: npt.ArrayLike,
    points: npt.ArrayLike,
) -> npt.NDArray[np.integer]:
    r"""
    Evaluates polynomials with coefficients in a finite field, using Horner's method. Coefficients and points are given by their integer representations, in $[0 : 2^k)$.

    Parameters:
        field: A finite field.
        coefficients: The coefficients of the polynomials, in increasing order of degree along the last dimension.
        points: The points at which to evaluate the polynomials. Must be a 1D-array.

    Returns:
        values: The values of the polynomials at the points. Has the same shape as `coefficients`, but with the last dimension replaced by the number of points.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> coefficients = [1, 1, 0, 0b0110]  # 1 + X + α^5 X^3
        >>> points = [0, 1, 0b0111, 0b1000, 0b1111]  # 0, 1, α^10, α^3, α^12
        >>> horner(field, coefficients, points)
        array([1, 6, 0, 0, 0])
    """
    coefficients, points = np.asarray(coefficients), np.asarray(points)
    values = np.zeros(coefficients.shape[:-1] + points.shape, dtype=int)
    for i in reversed(range(coefficients.shape[-1])):
        values = multiply(field, values, points) ^ coefficients[..., i, np.newaxis]
    return values


def convolve(
    field: FiniteBifield,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> npt.NDArray[np.integer]:
    r"""
    Multiplies polynomials with coefficients in a finite field. Coefficients are given by their integer representations, in $[0 : 2^k)$, in increasing order of degree along the last dimension. The other dimensions are broadcast.

    Parameters:
        field: A finite field.
        x: The coefficients of the first factor.
        y: The coefficients of the second factor.

    Returns:
        product: The coefficients of the product. Its last dimension has length `x.shape[-1] + y.shape[-1] - 1`.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> convolve(field, [0b0010, 1], [0b0100, 1])  # (α + X)(α^2 + X)
        array([8, 6, 1])
    """
    x, y = np.asarray(x), np.asarray(y)
    shape = np.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    product = np.zeros(shape + (x.shape[-1] + y.shape[-1] - 1,), dtype=int)
    for i in range(y.shape[-1]):
        product[..., i : i + x.shape[-1]] ^= multiply(field, x, y[..., i, np.newaxis])
    return product


def deconvolve(
    field: FiniteBifield,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    r"""
    Divides polynomials with coefficients in a finite field. Coefficients are given by their integer representations, in $[0 : 2^k)$, in increasing order of degree along the last dimension. The other dimensions are broadcast.

    Parameters:
        field: A finite field.
        x: The coefficients of the dividend.
        y: The coefficients of the divisor. The last one must be nonzero.

    Returns:
        quotient: The coefficients of the quotient. Its last dimension has length `x.shape[-1] - y.shape[-1] + 1`.
        remainder: The coefficients of the remainder. Its last dimension has length `y.shape[-1] - 1`.

    Raises:
        ZeroDivisionError: If the last coefficient of the divisor is zero.

    Examples:
        >>> field = komm.FiniteBifield(4)
        >>> x = [0b1001, 0b0110, 1]  # α^14 + α^5 X + X^2
        >>> y = [0b0010, 1]  # α + X
        >>> deconvolve(field, x, y)
        (array([4, 1]), array([1]))
    """
    x, y = np.asarray(x), np.asarray(y)
    shape = np.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    d = y.shape[-1] - 1  # Degree of the divisor.
    remainder = np.broadcast_to(x, shape + x.shape[-1:]).astype(int)
    quotient = np.zeros(shape + (x.shape[-1] - d,), dtype=int)
    for i in reversed(range(quotient.shape[-1])):
        q = divide(field, remainder[..., i + d], y[..., d])
        remainder[..., i : i + d + 1] ^= multiply(field, q[..., np.newaxis], y)
        quotient[..., i] = q
    return quotient, remainder[..., :d]


def binary_matrix(
    field: FiniteBifield,
    matrix: npt.ArrayLike,
) -> npt.NDArray[np.integer]:
    r"""
    Computes the binary matrix of a matrix over a finite field. Entries are given by their integer representations, in $[0 : 2^k)$.

    The binary matrix of an $r \times c$ matrix $A$ is the $kr \times kc$ binary matrix $B$ such that $\phi(a A) = \phi(a) B$ for every row vector $a$ of length $r$, where $\phi$ replaces each entry by its $k$ bits, in LSB-first order (that is, its coordinates with respect to the polynomial basis). In particular, if $A$ is a generator matrix of a code, then $B$ is a generator matrix of its binary image.

    Parameters:
        field: A finite field.
        matrix: The matrix $A$. Must be a 2D-array.

    Returns:
        binary_matrix: The binary matrix $B$.

    Examples:
        >>> field = komm.FiniteBifield(3)
        >>> binary_matrix(field, [[0b010]])  # α
        array([[0, 1, 0],
               [0, 0, 1],
               [1, 1, 0]])

        >>> field = komm.FiniteBifield(2)
        >>> binary_matrix(field, [[0b10, 1, 0], [0b11, 0, 1]])  # [[α, 1, 0], [α^2, 0, 1]]
        array([[0, 1, 1, 0, 0, 0],
               [1, 1, 0, 1, 0, 0],
               [1, 1, 0, 0, 1, 0],
               [1, 0, 0, 0, 0, 1]])
    """
    matrix = np.asarray(matrix)
    basis = 1 << np.arange(field.degree)
    products = multiply(field, matrix[:, np.newaxis], basis[:, np.newaxis])
    bits = int_to_bits(products, width=field.degree)
    return bits.reshape(-1, bits.shape[-1])
