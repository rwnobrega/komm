# BER of Viterbi decoding

The bit error rate (BER) of a convolutional code with soft-decision Viterbi decoding over the additive white Gaussian noise (AWGN) channel has no exact formula, but the union bound gives an upper bound, tight for high $E_\mathrm{b} / N_0$. This page compares the bound with a simulation, for the $(3, 1, 2)$ [convolutional code](/ref/ConvolutionalCode) with generator matrix
$$
    G(D) =
    \begin{bmatrix}
        1 + D  &&  1 + D^2  &&  1 + D + D^2
    \end{bmatrix}
$$
and free distance $d_\mathrm{free} = 7$ <cite>LC04, Example 12.5</cite>.

```pycon
>>> import numpy as np
>>> import komm

>>> code = komm.ConvolutionalCode([[0b011, 0b101, 0b111]])
>>> code.free_distance()
7

```

The union bound on the BER is <cite>LC04, eq. (12.44b)</cite>
$$
    P_\mathrm{b} \leq \frac{1}{k} \sum_j B_j \, \mathrm{Q} \left( \sqrt{2 j R E_\mathrm{b} / N_0} \right),
$$
where $\mathrm{Q}$ is the [Gaussian Q-function](/ref/gaussian_q), $R$ is the rate of the code, and $B_j$ is the total input weight of the paths that leave the all-zero state and first return to it with output weight $j$. (The book puts the factor $1 / k$ inside $B_j$.) Here, $k = 1$. The [path weight distribution](/ref/ConvolutionalCode#path_weight_distribution) of the code is the table $A_{i,j}$ of the number of these paths with input weight $i$ and output weight $j$, so that $B_j = \sum_i i A_{i,j}$. The bound is a series, which diverges below about $0.6$ dB. With the terms up to $j = 60$, its sum is within $0.2\%$ of the limit from $2$ dB on.

```pycon
>>> table = code.path_weight_distribution(60)
>>> B = np.arange(len(table)) @ table
>>> B[7:11]
array([1, 2, 3, 6])

```

The simulation uses the [terminated convolutional code](/ref/TerminatedConvolutionalCode) with $h = 1000$ information blocks and zero termination. Its rate, $R = 1000 / 3006$, is a little below $1 / 3$, since the tail carries no information. Each coded bit $v$ is sent as $(-1)^v$, with energy $1$, so $E_\mathrm{b} = 1 / R$. As in [BER of binary antipodal signaling](ber-antipodal.md), the [Gaussian channel](/ref/GaussianChannel) has noise power $N_0 / 2$, and a [discrete memoryless source](/ref/DiscreteMemorylessSource) emits the bits. The [Viterbi decoder](/ref/ViterbiDecoder) takes [L-values](/conventions#l-values), which are $4 r / N_0$, as in [Hard- and soft-decision decoding](hard-soft-decoding.md). Each value of $E_\mathrm{b} / N_0$ simulates batches of $100$ codewords until it counts at least $100$ bit errors, enough to keep the relative error of each estimate at most about $10\%$.

```pycon
>>> rng = np.random.default_rng(seed=42)
>>> komm.global_rng.set(rng)

>>> terminated = komm.TerminatedConvolutionalCode(code, num_blocks=1000)
>>> decoder = komm.ViterbiDecoder(terminated, input_type="soft")
>>> source = komm.DiscreteMemorylessSource(2)
>>> rate = terminated.rate
>>> eb = 1 / rate
>>> ebn0_db = np.arange(11)
>>> ebn0 = 10 ** (ebn0_db / 10)
>>> ber = []
>>> for n0 in eb / ebn0[:7]:
...     channel = komm.GaussianChannel(n0 / 2)
...     num_errors = num_bits = 0
...     while num_errors < 100:
...         bits = source.emit((100, terminated.dimension))
...         received = channel.transmit((-1.0) ** terminated.encode(bits))
...         bits_hat = decoder.decode(4 * received / n0)
...         num_errors += np.sum(bits != bits_hat)
...         num_bits += bits.size
...     ber.append(num_errors / num_bits)

```

The bound goes from $2$ to $7$ dB, with the rate of the terminated code. Without coding, $P_\mathrm{b} = \mathrm{Q}(\sqrt{2 E_\mathrm{b} / N_0})$.

```pycon
>>> def pairwise(j):
...     return komm.gaussian_q(np.sqrt(2 * j * rate * ebn0[2:8]))

>>> bound = sum(B[j] * pairwise(j) for j in range(len(B)))
>>> uncoded = komm.gaussian_q(np.sqrt(2 * ebn0))

```

To plot the curves, with [Matplotlib](https://matplotlib.org):

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(ebn0_db, uncoded, "k--", label="Uncoded")
ax.semilogy(ebn0_db[2:8], bound, "C0", label="Union bound")
ax.semilogy(ebn0_db[:7], ber, "C0o", label="Simulation")
ax.set(xlabel=r"$E_\mathrm{b}/N_0$ (dB)", ylabel="BER", ylim=(1e-6, 1e-1))
ax.set_xticks(ebn0_db)
ax.grid(which="both")
ax.legend(loc="lower left")
plt.show()
```

<figure markdown>
  ![Bit error rate of soft-decision Viterbi decoding of the (3, 1, 2) convolutional code over the AWGN channel, by simulation and by the union bound, compared with uncoded transmission.](/fig/ber_viterbi.svg)
</figure>

The union bound is loose for low $E_\mathrm{b} / N_0$, where it even exceeds the BER of uncoded transmission, and tight for high $E_\mathrm{b} / N_0$. At $6$ dB, the simulation lands on the bound, within its statistical error. For a BER of $10^{-5}$, the code needs about $6.2$ dB, by the bound, against $9.6$ dB without coding, a gain of about $3.4$ dB. For high $E_\mathrm{b} / N_0$, the gain approaches $10 \log_{10} (R d_\mathrm{free})$ dB, which is about $3.7$ dB for $R = 1 / 3$ and $d_\mathrm{free} = 7$.
