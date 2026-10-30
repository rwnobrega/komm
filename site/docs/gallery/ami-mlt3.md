# AMI and MLT-3 line codes

A _line code_ maps bits to the levels of a baseband waveform. In _alternate mark inversion_ (AMI), a $0$ gives the level $0$, and the $1$'s give $+1$ and $-1$, alternately. In _multilevel transmit 3_ (MLT-3), a $0$ keeps the level, and a $1$ moves it to the next one in the cycle $0, +1, 0, -1$. AMI is used in T1 lines, and MLT-3 in Fast Ethernet (100BASE-TX).

Both codes can be modeled as [Moore machines](/ref/MooreMachine) with the same four states: state $1$ has level $+1$, state $3$ has level $-1$, and states $0$ and $2$ have level $0$, with $+1$ and $-1$, respectively, as the next nonzero level. They differ only at the nonzero levels: there, with a $0$, AMI returns to zero and MLT-3 stays; with a $1$, AMI flips the sign and MLT-3 returns to zero. Their state diagrams are depicted below, AMI on the left and MLT-3 on the right.

<div class="centered" markdown>
  <span>
  ![State diagram of AMI.](/fig/machine_ami.svg)
  </span>
  <span>
  ![State diagram of MLT-3.](/fig/machine_mlt3.svg)
  </span>
</div>

The outputs $0$, $1$, and $2$ of the machines stand for the levels $0$, $+1$, and $-1$, respectively:

```pycon
>>> import numpy as np
>>> import komm

>>> ami = komm.MooreMachine(
...     transitions=[[0, 1], [2, 3], [2, 3], [0, 1]],
...     outputs=[0, 1, 0, 2],
... )
>>> mlt3 = komm.MooreMachine(
...     transitions=[[0, 1], [1, 2], [2, 3], [3, 0]],
...     outputs=[0, 1, 0, 2],
... )
>>> def encode(machine, bits):
...     output, _ = machine.process(bits, initial_state=0)
...     return np.array([0, +1, -1])[output]

>>> bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0])
>>> u_ami = encode(ami, bits)
>>> u_ami
array([ 0,  1, -1,  0,  1,  0,  0, -1,  1, -1,  0,  1, -1,  0,  0,  0])
>>> u_mlt3 = encode(mlt3, bits)
>>> u_mlt3
array([ 0,  1,  0,  0, -1, -1, -1,  0,  1,  0,  0, -1,  0,  0,  0,  0])

```

To plot the waveforms with the NRZ pulse, with [Matplotlib](https://matplotlib.org), and polar NRZ ($0 \mapsto -1$, $1 \mapsto +1$) for comparison:

```python
import matplotlib.pyplot as plt

examples = {"Polar NRZ": 2 * bits - 1, "AMI": u_ami, "MLT-3": u_mlt3}

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
for ax, name in zip(axs, examples):
    ax.stairs(examples[name], baseline=None, lw=2)
    ax.set(ylabel=name, xticks=range(bits.size + 1), yticks=[-1, 0, 1])
    ax.set_axisbelow(True)
    ax.grid()
axs[-1].set(xlabel="$t$")
plt.show()
```

<figure markdown>
  ![Waveforms of polar NRZ, AMI, and MLT-3.](/fig/ami_mlt3.svg)
</figure>

For random bits, a line code with levels $u_n$ and pulse $p(t)$, with time measured in bit periods, has power spectral density $S(f) = |\hat{p}(f)|^2 S_u(f)$, where $S_u$ is the DTFT of the autocorrelation $R[\ell] = \mathrm{E}[u_n u_{n+\ell}]$ of the levels <cite>PS08, Sec. 3.4–2</cite>. For polar NRZ, the levels are uncorrelated, so that $S_u(\phi) = 1$. For AMI, $R[0] = 1/2$, $R[\pm 1] = -1/4$, and $R[\ell] = 0$ otherwise, so that
$$
    S_u(\phi) = \sin^2(\pi \phi),
$$
with no power at DC. For MLT-3, $R[\ell] = 2^{-|\ell|/2 - 1} \cos(\pi \ell / 4)$, a damped cosine, and
$$
    S_u(\phi) = \frac{3 - 2 \cos(2 \pi \phi)}{10 - 24 \cos(2 \pi \phi) + 16 \cos^2(2 \pi \phi)},
$$
which peaks near $\phi = 1/8$: half of the bits are ones, so the level takes eight bits, on average, to go around the cycle.

The simulation below estimates the power spectral densities by averaging the periodograms of $1000$ blocks of $64$ bits:

```pycon
>>> komm.global_rng.set(np.random.default_rng(seed=42))

>>> source = komm.DiscreteMemorylessSource(2)
>>> bits = source.emit(64_000)

>>> sps = 8
>>> pulse = komm.RectangularPulse()  # NRZ pulse
>>> taps = pulse.taps(samples_per_symbol=sps)
>>> def periodogram(levels):
...     blocks = komm.sampling_rate_expand(levels.reshape(-1, 64), factor=sps)
...     waveform = komm.convolve(blocks, taps)
...     spectrum, freqs = komm.fourier_transform(waveform, time_step=1 / sps)
...     return freqs, np.mean(np.abs(spectrum) ** 2, axis=0) / 64

>>> estimates = {
...     "Polar NRZ": periodogram(2 * bits - 1),
...     "AMI": periodogram(encode(ami, bits)),
...     "MLT-3": periodogram(encode(mlt3, bits)),
... }

```

To plot the estimates, and the formulas as dashed lines:

```python
f = np.linspace(0, 2, 401)
c = np.cos(2 * np.pi * f)
esd = pulse.energy_spectral_density(f)
formulas = {
    "Polar NRZ": (f, esd),
    "AMI": (f, esd * np.sin(np.pi * f) ** 2),
    "MLT-3": (f, esd * (3 - 2 * c) / (10 - 24 * c + 16 * c**2)),
}

fig, ax = plt.subplots()
for name in estimates:
    ax.plot(*estimates[name], label=name)
    ax.plot(*formulas[name], "k--", lw=1)
ax.set(xlabel="$f$", ylabel="$S(f)$", xlim=(0, 2), ylim=(0, 1.6))
ax.grid()
ax.legend()
plt.show()
```

<figure markdown>
  ![Power spectral densities of polar NRZ, AMI, and MLT-3.](/fig/ami_mlt3_psd.svg)
</figure>

MLT-3 keeps $80\%$ of its power below a quarter of the bit rate, against $47\%$ for polar NRZ and $16\%$ for AMI. In 100BASE-TX, with $125$ Mbit/s on the line, that is $31.25$ MHz. The price is the power at DC, which AMI avoids.
