
# coding: utf-8

# # Binary sequences

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import matplotlib.pylab as plt
import komm


# ## Barker sequence

# In[2]:


barker = komm.BarkerSequence(length=13)
fig = plt.figure()
ax = plt.axes()
ax.stem(np.arange(barker.length), barker.polar_sequence)
ax.set_title(repr(barker))
fig.show()


# ## Walsh-Hadamard sequence

# In[3]:


walsh_hadamard = komm.WalshHadamardSequence(length=32, index=29)
fig = plt.figure()
ax = plt.axes()
ax.stem(np.arange(walsh_hadamard.length), walsh_hadamard.polar_sequence)
ax.set_title(repr(walsh_hadamard))
fig.show()


# ## Linear-feedback shift register (LFSR) sequence

# In[4]:


lfsr = komm.LFSRSequence(0b100101)
shifts = np.arange(-2*lfsr.length + 1, 2*lfsr.length)
cyclic_acorr = lfsr.cyclic_autocorrelation(shifts, normalized=True)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

ax0.stem(range(lfsr.length), lfsr.polar_sequence)
ax0.set_title(repr(lfsr))

ax1.plot(shifts, cyclic_acorr)
ax1.set_title('Cyclic autocorrelation')
ax1.grid(True)

fig.show()

