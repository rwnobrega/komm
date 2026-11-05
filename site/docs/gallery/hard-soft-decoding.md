# Hard- and soft-decision decoding

With binary antipodal signaling, the codewords of an $(n, k)$ [linear block code](/ref/BlockCode) become the points of a [constellation](/ref/Constellation) of dimension $n$ and order $M = 2^k$. The [PAM constellation](/ref/PAMConstellation) of order $2$ sends each bit $0$ as $-1$ and each bit $1$ as $+1$, so that the codeword $v$ becomes the point $x = 2v - 1$. For the $(7, 4)$ [Hamming code](/ref/HammingCode), the constellation has dimension $7$ and order $M = 16$. Row $i$ of the [codewords](/ref/BlockCode#codewords) of the code is the codeword of message $i$, so the index of a point is its message.

```pycon
>>> import numpy as np
>>> import komm

>>> code = komm.HammingCode(3)
>>> n, k = code.length, code.dimension
>>> M = 2**k
>>> pam2 = komm.PAMConstellation(2)
>>> points = pam2.indices_to_symbols(code.codewords())
>>> constellation = komm.Constellation(points)

```

Over the additive white Gaussian noise (AWGN) channel, the maximum-likelihood decoder, optimal for equiprobable messages, returns the point closest to the received word $r$ in Euclidean distance. This is *soft-decision* decoding (SDD). In *hard-decision* decoding (HDD), the receiver first decides each bit on its own, which gives a binary word $b$, and then returns the codeword closest to $b$ in Hamming distance. Since the point of $b$ is $h = 2b - 1$ and $\lVert h - x \rVert^2 = 4 \, \mathrm{d_H}(b, v)$, this is the point closest to $h$, so that the two decoders differ only in their input. The hard decisions are, coordinate by coordinate, the closest points of the PAM constellation. For example, the received word below is close to the point of message $0$, but two of its coordinates are slightly positive:

```pycon
>>> r = np.array([-0.8, 0.2, 0.1, -1.1, -0.9, -1.2, -1.0])
>>> b = pam2.closest_indices(r)
>>> b
array([0, 1, 1, 0, 0, 0, 0])
>>> h = pam2.indices_to_symbols(b)
>>> h
array([-1.,  1.,  1., -1., -1., -1., -1.])
>>> constellation.closest_indices(h)
array([7])
>>> constellation.closest_indices(r)
array([0])

```

The word $b$ has two bit errors, more than the Hamming code corrects, so hard-decision decoding returns a wrong message. Soft-decision decoding still sees that the two positive coordinates are small, and returns the correct message $0$.

The [exhaustive codeword decoder](/ref/ExhaustiveCodewordDecoder) makes the same decisions by another computation. All points have energy $n$, so that $\lVert r - x \rVert^2 = \lVert r \rVert^2 - 2 \langle r, x \rangle + n$, and the closest point is the one of largest correlation $\langle r, x \rangle$, which is what that decoder maximizes. For hard-decision decoding, it takes the decided bits $b$. For soft-decision decoding, it takes L-values. Over the AWGN channel and with the PAM constellation of order $2$, they are $-4 r / N_0$, since bit $0$ is sent as $-1$. With $N_0 = 1$, for example (any $N_0$ gives the same decisions), the two decoders return the bits of messages $7$ and $0$, least significant first:

```pycon
>>> hdd_decoder = komm.ExhaustiveCodewordDecoder(code, input_type="hard")
>>> hdd_decoder.decode(b)
array([1, 1, 1, 0])

>>> sdd_decoder = komm.ExhaustiveCodewordDecoder(code, input_type="soft")
>>> n0 = 1.0
>>> l_values = -4 * r / n0
>>> sdd_decoder.decode(l_values)
array([0, 0, 0, 0])

```

The simulation below estimates the block error rate (BLER) of the two decoders. Each point carries $k = 4$ bits, so $E_\mathrm{b}$ is the mean energy of the constellation divided by $k$. As in [BER of binary antipodal signaling](ber-antipodal.md), the signal is real, so the [Gaussian channel](/ref/GaussianChannel) has noise power $N_0 / 2$, and a [discrete memoryless source](/ref/DiscreteMemorylessSource), here of cardinality $M$, emits the messages. Each value of $E_\mathrm{b} / N_0$ simulates batches of $10^4$ codewords until soft-decision decoding, which errs less, counts at least $100$ block errors, enough to keep the relative error of each estimate at most about $10\%$.

```pycon
>>> rng = np.random.default_rng(seed=42)
>>> komm.global_rng.set(rng)

>>> source = komm.DiscreteMemorylessSource(M)
>>> eb = constellation.mean_energy() / k
>>> ebn0_db = np.arange(8)
>>> ebn0 = 10 ** (ebn0_db / 10)
>>> bler_hdd, bler_sdd = [], []
>>> for n0 in eb / ebn0:
...     channel = komm.GaussianChannel(n0 / 2)
...     num_errors_hdd = num_errors_sdd = num_blocks = 0
...     while num_errors_sdd < 100:
...         messages = source.emit(10_000)
...         symbols = constellation.indices_to_symbols(messages)
...         received = channel.transmit(symbols)
...         hard = pam2.closest_symbols(received)
...         messages_hdd = constellation.closest_indices(hard)
...         messages_sdd = constellation.closest_indices(received)
...         num_errors_hdd += np.sum(messages != messages_hdd)
...         num_errors_sdd += np.sum(messages != messages_sdd)
...         num_blocks += messages.size
...     bler_hdd.append(num_errors_hdd / num_blocks)
...     bler_sdd.append(num_errors_sdd / num_blocks)

```

For hard-decision decoding, the BLER has an exact formula. The hard decisions turn the AWGN channel into a [binary symmetric channel](/ref/BinarySymmetricChannel) with crossover probability
$$
    p = \mathrm{Q} \left( \sqrt{2 R E_\mathrm{b} / N_0} \right),
$$
where $\mathrm{Q}$ is the [Gaussian Q-function](/ref/gaussian_q) and $R = k / n$ is the rate of the code, since each coded bit has energy $R E_\mathrm{b}$. As in [BER of syndrome decoding](ber-syndrome.md),
$$
    P_\mathrm{block} = 1 - \sum_{w = 0}^{n} \alpha_w p^w q^{n - w},
$$
where $q = 1 - p$ and $\alpha$ is the [coset leader weight distribution](/ref/BlockCode#coset_leader_weight_distribution) of the code. For the Hamming code, $\alpha_0 = 1$ and $\alpha_1 = 7$.

```pycon
>>> rate = code.rate
>>> p = komm.gaussian_q(np.sqrt(2 * rate * ebn0))
>>> q = 1 - p
>>> alpha = code.coset_leader_weight_distribution()
>>> hdd_formula = 1 - sum(alpha[w] * p**w * q ** (n - w) for w in range(n + 1))

```

For soft-decision decoding, there is no such formula, but there is the union bound <cite>LC04, Sec. 10.1</cite>
$$
    P_\mathrm{block} \leq \sum_{w \geq 1} A_w \, \mathrm{Q} \left( \sqrt{2 w R E_\mathrm{b} / N_0} \right),
$$
where $A_w$ is the number of codewords of weight $w$, given by the [codeword weight distribution](/ref/BlockCode#codeword_weight_distribution) of the code. For the Hamming code, $A_3 = A_4 = 7$ and $A_7 = 1$. Without coding, a block of $k$ bits has $P_\mathrm{block} = 1 - (1 - \mathrm{Q}(\sqrt{2 E_\mathrm{b} / N_0}))^k$.

```pycon
>>> def pairwise(w):
...     return komm.gaussian_q(np.sqrt(2 * w * rate * ebn0))

>>> A = code.codeword_weight_distribution()
>>> sdd_bound = sum(A[w] * pairwise(w) for w in range(1, n + 1))
>>> uncoded = 1 - (1 - komm.gaussian_q(np.sqrt(2 * ebn0))) ** k

```

To plot the curves, with [Matplotlib](https://matplotlib.org):

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(ebn0_db, uncoded, "k--", label="Uncoded")
ax.semilogy(ebn0_db, hdd_formula, "C0", label="Hard, formula")
ax.semilogy(ebn0_db, bler_hdd, "C0o", label="Hard, simulation")
ax.semilogy(ebn0_db, sdd_bound, "C1", label="Soft, union bound")
ax.semilogy(ebn0_db, bler_sdd, "C1s", label="Soft, simulation")
ax.set(xlabel=r"$E_\mathrm{b}/N_0$ (dB)", ylabel="BLER", ylim=(1e-4, 1))
ax.set_xticks(ebn0_db)
ax.grid(which="both")
ax.legend(loc="lower left")
plt.show()
```

<figure markdown>
  ![Block error rate of hard- and soft-decision decoding of the (7, 4) Hamming code over the AWGN channel, compared with uncoded transmission.](/fig/hard_soft_decoding.svg)
</figure>

The simulation follows the formula for hard-decision decoding. The union bound is loose for low $E_\mathrm{b} / N_0$, where it even exceeds the BLER of uncoded transmission, and tight for high $E_\mathrm{b} / N_0$. For a BLER of $10^{-2}$, hard-decision decoding needs about $5.5$ dB, only $0.5$ dB less than uncoded transmission, while soft-decision decoding needs about $4.3$ dB, by the union bound, a gain of about $1.2$ dB. It can be shown that, for high $E_\mathrm{b} / N_0$, the gain of soft decisions approaches $10 \log_{10} (d / (t + 1))$ dB, where $d$ is the minimum distance of the code and $t = \lfloor (d - 1) / 2 \rfloor$. Here, $d = 3$ and $t = 1$, which gives $10 \log_{10} (3 / 2) \approx 1.8$ dB.
