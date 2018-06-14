
# coding: utf-8

# # Binary sequences

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pylab as plt
import ipywidgets
import komm


# ## Barker sequence

# In[2]:


def barker_demo(index):
    length = [2, 3, 4, 5, 7, 11, 13][index]
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

ipywidgets.interact(barker_demo, index=(0, 6));


# ## Walsh-Hadamard sequence

# In[3]:


log_length_widget = ipywidgets.IntSlider(min=0, max=7, step=1, value=3)
index_widget = ipywidgets.IntSlider(min=0, max=7, step=1, value=0)

def update_index_widget(*args):
    index_widget.max = 2 ** log_length_widget.value - 1
log_length_widget.observe(update_index_widget, 'value')

def walsh_hadamard_demo(log_length, ordering, index):
    length = 2**log_length
    walsh_hadamard = komm.WalshHadamardSequence(length=length, ordering=ordering, index=index)
    ax = plt.axes()
    ax.stem(np.arange(length), walsh_hadamard.polar_sequence)
    ax.set_title(repr(walsh_hadamard))
    ax.set_xlabel('$n$')
    ax.set_ylabel('$a[n]$')
    ax.set_yticks([-1, 0, 1])
    ax.set_ylim([-1.2, 1.2])
    plt.show()
    
ipywidgets.interact(walsh_hadamard_demo, log_length=log_length_widget, ordering=['natural', 'sequency'], index=index_widget);


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

