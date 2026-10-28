# BER of syndrome decoding

Over the [binary symmetric channel](/ref/BinarySymmetricChannel) with crossover probability $p$, the bit error rate (BER) of syndrome decoding has an exact formula in terms of the [standard array](standard-array.md) of the [linear block code](/ref/BlockCode). By linearity, the bit errors do not depend on the transmitted message, so assume that it is the all-zero message. Then a bit error at position $i$ occurs if and only if $\hat{u}\_i = 1$, where $\hat{u}$ is the decoded message. The received word is the error pattern $e$. If $e$ falls in the column $\mathrm{Col}(u)$ of the array, headed by the codeword of message $u$, the decoder returns that codeword, so that $\hat{u} = u$. Therefore <cite>MS77, Ch. 1, Problem 23</cite>,
$$
\begin{aligned}
    P_\mathrm{b} & = \frac{1}{k} \sum_i \Pr[\hat{u}\_i = 1] \\\\
    & = \frac{1}{k} \sum_u \mathrm{w_H}(u) \Pr[e \in \mathrm{Col}(u)],
\end{aligned}
$$
where an error pattern of weight $w$ has probability $p^w q^{n - w}$, with $q = 1 - p$. The column $\mathrm{Col}(0)$ of the all-zero message holds the coset leaders, the only error patterns that the decoder corrects, so that the block error probability is $P_\mathrm{block} = 1 - \Pr[e \in \mathrm{Col}(0)]$.

Both probabilities are thus polynomials in $p$:
$$
    P_\mathrm{block} = 1 - \sum_w \alpha_w p^w q^{n - w},
    \qquad
    P_\mathrm{b} = \sum_w \beta_w p^w q^{n - w},
$$
where $\alpha_w$ is the number of words of weight $w$ in $\mathrm{Col}(0)$, and $\beta_w$ is the sum, over all messages $u$, of $\mathrm{w_H}(u) / k$ times the number of words of weight $w$ in $\mathrm{Col}(u)$. The function `coefficients` below returns $\alpha$ and $\beta$, and the function `evaluate` computes such a sum:

```pycon
>>> import numpy as np
>>> import komm

>>> def coefficients(code):
...     n, k = code.length, code.dimension
...     array = code.coset_leaders()[:, np.newaxis] ^ code.codewords()
...     counts = [np.bincount(col, minlength=n + 1) for col in array.sum(-1).T]
...     message_weights = code.inverse_encode(code.codewords()).sum(axis=-1)
...     return counts[0], message_weights @ counts / k

>>> def evaluate(c, p):
...     n = len(c) - 1
...     w = np.arange(n + 1)
...     p = np.asarray(p)[..., np.newaxis]
...     return np.sum(c * p**w * (1 - p) ** (n - w), axis=-1)

```

For the [Hamming code](/ref/HammingCode) of length $7$:

```pycon
>>> code = komm.HammingCode(3)
>>> alpha, beta = coefficients(code)
>>> alpha
array([1, 7, 0, 0, 0, 0, 0, 0])
>>> beta
array([ 0.,  0.,  9., 19., 16., 12.,  7.,  1.])

```

Here, $\alpha$ is the [coset leader weight distribution](/ref/BlockCode#coset_leader_weight_distribution) of the code, and $\beta$ gives
$$
\begin{aligned}
    P_\mathrm{b} & = 9p^2 q^5 + 19p^3 q^4 + 16p^4 q^3 + 12p^5 q^2 + 7p^6 q + p^7 \\\\
    & = 9p^2 - 26p^3 + 30p^4 - 12p^5.
\end{aligned}
$$

The simulation below confirms this formula. The message bits come from a [discrete memoryless source](/ref/DiscreteMemorylessSource), and the receiver uses the [syndrome table decoder](/ref/SyndromeTableDecoder). For comparison, it also uses the [bitwise MAP decoder](/ref/ExhaustiveBitwiseDecoder), which minimizes the BER, with the L-value $(-1)^{r_i} \log(q / p)$ of each received bit $r_i$.

```pycon
>>> rng = np.random.default_rng(seed=42)
>>> komm.global_rng.set(rng)

>>> source = komm.DiscreteMemorylessSource(2)
>>> decoder = komm.SyndromeTableDecoder(code)
>>> bitwise = komm.ExhaustiveBitwiseDecoder(code, output_type="hard")
>>> ps = np.linspace(0.05, 0.45, 9)
>>> ber, ber_bitwise = [], []
>>> for p in ps:
...     channel = komm.BinarySymmetricChannel(p)
...     bits = source.emit(100_000)
...     received = channel.transmit(code.encode(bits))
...     bits_hat = decoder.decode(received)
...     ber.append(np.mean(bits != bits_hat))
...     l_values = np.log((1 - p) / p) * (-1.0) ** received
...     bits_hat = bitwise.decode(l_values)
...     ber_bitwise.append(np.mean(bits != bits_hat))

```

Since the encoder is systematic, reading the message bits off the received word gives BER $p$. Decoding does better only for $p < p^\star = (3 - \sqrt{3}) / 6 \approx 0.211$, because
$$
    P_\mathrm{b} - p = -p (1 - p) (1 - 2p) (6p^2 - 6p + 1).
$$
To plot the curves, with [Matplotlib](https://matplotlib.org):

```python
import matplotlib.pyplot as plt

ps_fine = np.linspace(0, 0.5, 101)
p_star = (3 - np.sqrt(3)) / 6
fig, ax = plt.subplots()
ax.plot(ps_fine, evaluate(beta, ps_fine), label="Theory")
ax.plot(ps, ber, "o", label="Simulation")
ax.plot(ps_fine, ps_fine, label="Uncoded")
ax.plot(ps, ber_bitwise, "s", mfc="none", ms=9, label="Bitwise MAP")
ax.plot(p_star, p_star, "k*", ms=9)
ax.set(xlabel="$p$", ylabel="BER", xlim=(0, 0.5), ylim=(0, 0.5))
ax.grid()
ax.legend()
plt.show()
```

<figure markdown>
  ![Bit error rate of syndrome decoding and of bitwise MAP decoding of the Hamming code of length 7 over the binary symmetric channel, compared with uncoded transmission.](/fig/ber_syndrome.svg)
</figure>

The bitwise MAP decoder makes the same decisions as syndrome decoding for $p < p^\star$, and returns the received message bits for $p > p^\star$. Its BER is thus the smaller of the two curves.
