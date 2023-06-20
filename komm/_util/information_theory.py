import numpy as np


def _entropy_base_e(pmf):
    # Assumptions:
    # - pmf is a 1D numpy array.
    # - pmf is a valid pmf.
    return -np.dot(pmf, np.log(pmf, where=(pmf > 0)))


def _entropy_base_2(pmf):
    # Assumptions: Same as _entropy_base_e.
    return -np.dot(pmf, np.log2(pmf, where=(pmf > 0)))


def _entropy(pmf, base=2.0):
    # Assumptions: Same as _entropy_base_e.
    if base == "e":
        return _entropy_base_e(pmf)
    elif base == 2:
        return _entropy_base_2(pmf)
    else:
        return _entropy_base_e(pmf) / np.log(base)


def entropy(pmf, base=2.0):
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

        >>> np.around(komm.entropy([1/4, 1/4, 1/4, 1/4]), decimals=6)
        2.0

        >>> np.around(komm.entropy([1/3, 1/3, 1/3], base=3.0), decimals=6)
        1.0

        >>> komm.entropy([1.0, 1.0])
        Traceback (most recent call last):
        ...
        ValueError: Invalid pmf
    """
    pmf = np.array(pmf, dtype=float)
    if not np.allclose(np.sum(pmf), 1.0) or not np.alltrue(pmf >= 0.0):
        raise ValueError("Invalid pmf")
    return _entropy(pmf, base)


def _mutual_information(input_pmf, transition_probabilities, base=2.0):
    output_pmf = np.dot(input_pmf, transition_probabilities)
    entropy_output_prior = _entropy(output_pmf, base=base)
    entropy_base = lambda pmf: _entropy(pmf, base=base)
    entropy_output_posterior = np.dot(input_pmf, np.apply_along_axis(entropy_base, 1, transition_probabilities))
    return entropy_output_prior - entropy_output_posterior
