
# coding: utf-8

# # Komm demo: Pulse formatting

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pylab as plt
import ipywidgets
import komm


# ## Sinc pulse (zero ISI)

# In[2]:


def sinc_demo(show_individual, show_signal):
    info = [1, -1, 1, 1, -1, -1, 1]
    pulse = komm.SincPulse(length_in_symbols=20)
    t0, t1 = pulse.interval
    tx_filter = komm.TransmitFilter(pulse, samples_per_symbol=32)
    signal = tx_filter(info)
    t = np.arange(t0, t1 + len(info) - 1, step=1/tx_filter.samples_per_symbol)

    _, ax = plt.subplots(figsize=(16, 10))
    if show_individual:
        for k, a in enumerate(info):
            ax.plot(t, a*pulse.impulse_response(t - k), 'k--')
    if show_signal:
        ax.plot(t, signal, 'b', linewidth=3)
    ax.stem(info, linefmt='r', markerfmt='ro')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$s(t)$')
    ax.set_xticks(np.arange(-2.0, 11.0))
    ax.set_xlim([-2.0, 10.0])
    ax.set_ylim([-1.75, 1.75])
    ax.grid()
    plt.show()

ipywidgets.interact(sinc_demo, show_individual=False, show_signal=False);


# ## Raised cosine pulse

# In[3]:


def raised_cosine_demo(rolloff):
    pulse = komm.RaisedCosinePulse(rolloff, length_in_symbols=20)
    h = pulse.impulse_response
    H = pulse.frequency_response
    t = np.linspace(-8.0, 8.0, 1000)
    f = np.linspace(-1.5, 1.5, 200)

    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    ax0.plot(t, h(t), 'b')
    ax0.axis([-7.1, 7.1, -.3, 1.1])
    ax0.set_title('Raised cosine pulse (waveform)')
    ax0.set_xlabel('$t$')
    ax0.set_ylabel('$h(t)$')
    ax0.grid()
    ax1.plot(f, H(f), 'r')
    ax1.axis([-1.1, 1.1, -.1, 1.1])
    ax1.set_title('Raised cosine pulse (spectrum)')
    ax1.set_xlabel('$f$')
    ax1.set_ylabel('$H(f)$')
    ax1.grid()
    plt.show()

rolloff_widget = ipywidgets.FloatSlider(min=0, max=1.0, step=0.1, value=0.5)

ipywidgets.interact(raised_cosine_demo, rolloff=rolloff_widget);


# ## Gaussian pulse

# In[4]:


def gaussian_pulse_demo(half_power_bandwidth):
    pulse = komm.GaussianPulse(half_power_bandwidth, length_in_symbols=4)
    h = pulse.impulse_response
    H = pulse.frequency_response
    t = np.linspace(-8.0, 8.0, 1000)
    f = np.linspace(-4.0, 4.0, 500)

    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    ax0.plot(t, h(t), 'b')
    ax0.axis([-7.1, 7.1, -.1, 1.1])
    ax0.set_title('Gaussian pulse (waveform)')
    ax0.set_xlabel('$t$')
    ax0.set_ylabel('$h(t)$')
    ax0.grid()
    ax1.plot(f, H(f), 'r')
    ax1.plot([-4.0, 4.0], [H(0) / np.sqrt(2), H(0) / np.sqrt(2)], linestyle='dashed', color='gray')
    ax1.plot([half_power_bandwidth, half_power_bandwidth], [-0.1*H(0), 1.1*H(0)], linestyle='dashed', color='gray')
    ax1.plot([-half_power_bandwidth, -half_power_bandwidth], [-0.1*H(0), 1.1*H(0)], linestyle='dashed', color='gray')
    ax1.axis([-2.0, 2.0, -0.1*H(0), 1.1*H(0)])
    ax1.set_title('Gaussian pulse (spectrum)')
    ax1.set_xlabel('$f$')
    ax1.set_ylabel('$H(f)$')
    ax1.grid()
    plt.show()

half_power_bandwidth_widget = ipywidgets.FloatSlider(min=0.05, max=1.0, step=0.01, value=0.5)

ipywidgets.interact(gaussian_pulse_demo, half_power_bandwidth=half_power_bandwidth_widget);

