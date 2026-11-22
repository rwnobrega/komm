# Conventions

Conventions used throughout Komm.

## L-values

The *L-value* of a bit $b$ is defined as
$$
    L(b) = \log \frac{\Pr[b = \mathtt{0}]}{\Pr[b = \mathtt{1}]},
$$
with the natural logarithm. Its sign gives the hard decision, positive for bit $\mathtt{0}$ and negative for bit $\mathtt{1}$; its magnitude gives the reliability of that decision.

Decoders with soft input take L-values, and decoders with soft output return them.

If $b_1$ and $b_2$ are independent, then $L(b_1 \oplus b_2) = L(b_1) \boxplus L(b_2)$, where $\oplus$ denotes the modulo-$2$ sum and $\boxplus$ the [box-plus operation](/ref/boxplus).

For equally likely bits, the L-value of a bit $b$ given the channel output $y$ is the logarithm of the likelihood ratio,
$$
    L(b \mid y) = \log \frac{p(y \mid b = \mathtt{0})}{p(y \mid b = \mathtt{1})}.
$$

- Over the [binary symmetric channel](/ref/BinarySymmetricChannel) with crossover probability $p$,
    $$
        L(b \mid y) = (-1)^y \log \frac{1 - p}{p}.
    $$

- Over the [binary erasure channel](/ref/BinaryErasureChannel), where $2$ denotes an erasure,
    $$
        L(b \mid y) = \begin{cases}
            +\infty, & y = \mathtt{0}, \\\\
            -\infty, & y = \mathtt{1}, \\\\
            0, & y = 2.
        \end{cases}
    $$

- Over the [Gaussian channel](/ref/GaussianChannel) with noise power $\sigma^2$, with each bit $b$ sent as $(-1)^b$,
    $$
        L(b \mid y) = 2 y / \sigma^2.
    $$
    This mapping is the [constellation](/ref/Constellation) with points $+1$ and $-1$, in this order, with the [natural labeling](/ref/NaturalLabeling), and the same L-values follow from the [posteriors](/ref/Constellation#posteriors) of the constellation, [marginalized](/ref/Labeling#marginalize) over the labeling:

    ```pycon
    >>> import numpy as np
    >>> import komm

    >>> constellation = komm.Constellation([1.0, -1.0])
    >>> labeling = komm.NaturalLabeling(1)
    >>> y = np.array([0.9, -0.2])
    >>> posteriors = constellation.posteriors(y, noise_power=0.5)
    >>> labeling.marginalize(posteriors)
    array([ 3.6, -0.8])
    >>> 2 * y / 0.5
    array([ 3.6, -0.8])

    ```
