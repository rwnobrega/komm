
# coding: utf-8

# # Komm demo: Binary sequences

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pylab as plt
import ipywidgets
import komm


# ## Barker sequence

# In[2]:


def barker_demo(length):
    barker = komm.BarkerSequence(length=length)
    shifts = np.arange(-2*length + 1, 2*length)
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.stem(np.arange(length), barker.polar_sequence)
    ax0.set_title(repr(barker))
    ax0.set_xlabel('$n$')
    ax0.set_ylabel('$a[n]$')
    ax0.set_xticks(np.arange(length))
    ax0.set_yticks([-1, 0, 1])
    ax1.stem(shifts, barker.autocorrelation(shifts))
    ax1.set_title('Autocorrelation')
    ax1.set_xlabel('$\\ell$')
    ax1.set_ylabel('$R[\\ell]$')
    ax1.set_xticks([-length, 0, length])
    ax1.set_yticks(np.arange(-1, length + 1))
    plt.show()

ipywidgets.interact(barker_demo, length=ipywidgets.SelectionSlider(options=[2, 3, 4, 5, 7, 11, 13]));


# ## Walsh-Hadamard sequence

# In[3]:


def walsh_hadamard_demo(length, ordering, index):
    walsh_hadamard = komm.WalshHadamardSequence(length=length, ordering=ordering, index=index)
    ax = plt.axes()
    ax.stem(np.arange(length), walsh_hadamard.polar_sequence)
    ax.set_title(repr(walsh_hadamard))
    ax.set_xlabel('$n$')
    ax.set_ylabel('$a[n]$')
    ax.set_yticks([-1, 0, 1])
    ax.set_ylim([-1.2, 1.2])
    plt.show()

length_widget = ipywidgets.SelectionSlider(options=[2**i for i in range(1, 8)])
index_widget = ipywidgets.IntSlider(min=0, max=7, step=1, value=0)

def update_index_widget(*args):
    index_widget.max = length_widget.value - 1
length_widget.observe(update_index_widget, 'value')

ipywidgets.interact(walsh_hadamard_demo, length=length_widget, ordering=['natural', 'sequency'], index=index_widget);


# ## Linear-feedback shift register (LFSR) sequence -- Maximum-length sequence (MLS)

# In[4]:


def lfsr_demo(degree):
    lfsr = komm.LFSRSequence.maximum_length_sequence(degree=degree)
    length = lfsr.length
    shifts = np.arange(-2*length + 1, 2*length)
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.stem(np.arange(length), lfsr.polar_sequence)
    ax0.set_title(repr(lfsr))
    ax0.set_xlabel('$n$')
    ax0.set_ylabel('$a[n]$')
    ax0.set_yticks([-1, 0, 1])
    ax1.stem(shifts, lfsr.cyclic_autocorrelation(shifts, normalized=True))
    ax1.set_title('Cyclic autocorrelation (normalized)')
    ax1.set_xlabel('$\\ell$')
    ax1.set_ylabel('$R[\\ell]$')
    ax1.set_xticks([-length, 0, length])
    ax1.set_ylim([-0.5, 1.1])
    plt.show()

ipywidgets.interact(lfsr_demo, degree=(2, 7));

