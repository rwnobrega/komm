# %% [markdown]
# # Komm demo: Binary sequences

# %%
import matplotlib.pylab as plt
import numpy as np

import komm

# %% [markdown]
# ## Barker sequence


# %%
def barker_demo(length):
    barker = komm.BarkerSequence(length=length)
    shifts = np.arange(-2 * length + 1, 2 * length)
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.stem(np.arange(length), barker.polar_sequence)
    ax0.set_title(repr(barker))
    ax0.set_xlabel("$n$")
    ax0.set_ylabel("$x[n]$")
    ax0.set_xticks(np.arange(length))
    ax0.set_yticks([-1, 0, 1])
    ax1.stem(shifts, barker.autocorrelation(shifts))
    ax1.set_title("Autocorrelation")
    ax1.set_xlabel("$\\ell$")
    ax1.set_ylabel("$R[\\ell]$")
    ax1.set_xticks([-length, 0, length])
    ax1.set_yticks(np.arange(-1, length + 1))
    plt.show()


barker_demo(13)

# %% [markdown]
# ## Walsh-Hadamard sequence


# %%
def walsh_hadamard_demo(length, ordering, index):
    walsh_hadamard = komm.WalshHadamardSequence(length=length, ordering=ordering, index=index)
    ax = plt.axes()
    ax.stem(np.arange(length), walsh_hadamard.polar_sequence)
    ax.set_title(repr(walsh_hadamard))
    ax.set_xlabel("$n$")
    ax.set_ylabel("$x[n]$")
    ax.set_yticks([-1, 0, 1])
    plt.show()


walsh_hadamard_demo(8, "natural", 6)


# %% [markdown]
# ## LFSR sequence


# %%
def lfsr_demo(degree):
    lfsr = komm.LFSRSequence.maximum_length_sequence(degree=degree)
    length = lfsr.length
    shifts = np.arange(-2 * length + 1, 2 * length)
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.stem(np.arange(length), lfsr.polar_sequence)
    ax0.set_title(repr(lfsr))
    ax0.set_xlabel("$n$")
    ax0.set_ylabel("$x[n]$")
    ax0.set_yticks([-1, 0, 1])
    ax1.stem(shifts, lfsr.cyclic_autocorrelation(shifts))
    ax1.set_title("Cyclic autocorrelation")
    ax1.set_xlabel("$\\ell$")
    ax1.set_ylabel("$R[\\ell]$")
    ax1.set_xticks([-length, 0, length])
    plt.show()


lfsr_demo(5)
