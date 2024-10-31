import numpy as np
from attrs import field

from .._validation import is_log_base, is_pmf, validate_call


def _entropy(pmf, base):
    # Assumptions:
    # - pmf is a valid pmf.
    def _entropy_base_e(pmf):
        return -np.dot(pmf, np.log(pmf, where=(pmf > 0)))

    if base == "e":
        return _entropy_base_e(pmf)
    elif base == 2.0:
        return -np.dot(pmf, np.log2(pmf, where=(pmf > 0)))
    else:
        return _entropy_base_e(pmf) / np.log(base)


@validate_call(
    pmf=field(converter=np.asarray, validator=is_pmf),
    base=field(validator=is_log_base),
)
def entropy(pmf, base: float | str = 2.0):
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

        >>> komm.entropy(base=3.0, pmf=[1/3, 1/3, 1/3])  # doctest: +NUMBER
        np.float64(1.0)
    """
    return _entropy(pmf, base)


def _mutual_information(input_pmf, transition_probabilities, base=2.0):
    output_pmf = np.dot(input_pmf, transition_probabilities)
    entropy_output_prior = _entropy(output_pmf, base=base)
    entropy_base = lambda pmf: _entropy(pmf, base=base)
    entropy_output_posterior = np.dot(input_pmf, np.apply_along_axis(entropy_base, 1, transition_probabilities))
    return entropy_output_prior - entropy_output_posterior
