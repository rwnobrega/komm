# BER of binary antipodal signaling

The bit error rate (BER) of binary antipodal signaling over the additive white Gaussian noise (AWGN) channel is
$$
    P_\mathrm{b} = \mathrm{Q} \left( \sqrt{2 E_\mathrm{b} / N_0} \right),
$$
where $\mathrm{Q}$ is the [Gaussian Q-function](/ref/gaussian_q). Binary phase-shift keying (BPSK) has the same BER. The simulation below confirms this formula.

Binary antipodal signaling uses the [PAM constellation](/ref/PAMConstellation) of order $2$, with points $\pm 1$, so that $E_\mathrm{b} = 1$. The [Gaussian channel](/ref/GaussianChannel) then has noise power $N_0 / 2 = 1 / (2 E_\mathrm{b} / N_0)$. The bits, from a [discrete memoryless source](/ref/DiscreteMemorylessSource), serve directly as indices of the constellation points, and the receiver picks the closest point.

```pycon
>>> import numpy as np
>>> import komm

>>> rng = np.random.default_rng(seed=42)
>>> komm.global_rng.set(rng)

>>> source = komm.DiscreteMemorylessSource(2)
>>> constellation = komm.PAMConstellation(2)
>>> ebn0_db = np.arange(9)
>>> ebn0 = 10 ** (ebn0_db / 10)
>>> ber = []
>>> for noise_power in 1 / (2 * ebn0):
...     channel = komm.GaussianChannel(noise_power)
...     bits = source.emit(1_000_000)
...     symbols = constellation.indices_to_symbols(bits)
...     received = channel.transmit(symbols)
...     bits_hat = constellation.closest_indices(received)
...     ber.append(np.mean(bits != bits_hat))

>>> ber_theory = komm.gaussian_q(np.sqrt(2 * ebn0))
>>> for row in zip(ebn0_db, ber, ber_theory):
...     print("{} dB  {:.2e}  {:.2e}".format(*row))
0 dB  7.85e-02  7.86e-02
1 dB  5.67e-02  5.63e-02
2 dB  3.74e-02  3.75e-02
3 dB  2.30e-02  2.29e-02
4 dB  1.24e-02  1.25e-02
5 dB  5.97e-03  5.95e-03
6 dB  2.35e-03  2.39e-03
7 dB  7.78e-04  7.73e-04
8 dB  1.89e-04  1.91e-04

```

To plot the curves, with [Matplotlib](https://matplotlib.org):

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(ebn0_db, ber_theory, label="Theory")
ax.semilogy(ebn0_db, ber, "o", label="Simulation")
ax.set(xlabel=r"$E_\mathrm{b}/N_0$ (dB)", ylabel="BER", ylim=(1e-4, 1e-1))
ax.grid(which="both")
ax.legend()
plt.show()
```

<figure markdown>
  ![Bit error rate of binary antipodal signaling over the AWGN channel.](/fig/ber_antipodal.svg)
</figure>
