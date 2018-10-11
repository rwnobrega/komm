
# coding: utf-8

# # Komm demo: Constellations

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pylab as plt
import ipywidgets
import komm


# In[2]:


def constellation_demo(modulation, noise_power_db, xlim, ylim):
    awgn = komm.AWGNChannel()

    num_symbols = 10000
    noise_power = 10**(noise_power_db / 10)
    awgn.signal_power = modulation.energy_per_symbol
    awgn.snr = awgn.signal_power / noise_power
    num_bits = modulation.bits_per_symbol * num_symbols
    bits = np.random.randint(2, size=num_bits)
    sentword = modulation.modulate(bits)
    recvword = awgn(sentword)

    _, ax = plt.subplots(figsize=(16, 10))
    ax.scatter(recvword.real, recvword.imag, color='xkcd:light blue', s=1)
    ax.scatter(modulation.constellation.real, modulation.constellation.imag, color='xkcd:blue', s=8**2)
    for (i, point) in enumerate(modulation.constellation):
        binary_label = ''.join(str(b) for b in komm.int2binlist(modulation.labeling[i], width=modulation.bits_per_symbol))
        ax.text(point.real, point.imag + 0.075 * xlim[0], binary_label, horizontalalignment='center')
    ax.set_title(repr(modulation))
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.axis('square')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()
    info_text = 'SNR = {:.1f} dB\n'.format(10*np.log10(awgn.snr))
    info_text += 'Eb/N0 = {:.1f} dB'.format(10*np.log10(awgn.snr / modulation.bits_per_symbol))
    ax.text(1.125 * xlim[1], 0.0, info_text, horizontalalignment='left', verticalalignment='center')
    plt.show()


# ## Phase-shift keying (PSK)

# In[3]:


def psk_constellation_demo(order, amplitude, phase_offset, labeling, noise_power_db):
    psk_modulation = komm.PSKModulation(order, amplitude, phase_offset, labeling)
    constellation_demo(psk_modulation, noise_power_db, xlim=[-3.0, 3.0], ylim=[-3.0, 3.0])

order_widget = ipywidgets.SelectionSlider(
    options=[2, 4, 8, 16, 32],
    continuous_update=False,
    description='Order:',
)

amplitude_widget = ipywidgets.FloatSlider(
    min=0.1,
    max=2.01,
    step=0.1,
    value=1.0,
    continuous_update=False,
    description='Amplitude:',
)

phase_offset_widget = ipywidgets.SelectionSlider(
    options=[('{:.2f}π'.format(x), np.pi*x) for x in np.arange(0.0, 2.001, step=0.01)],
    value=0.0,
    continuous_update=False,
    description='Phase offset:',
)

labeling_widget = ipywidgets.Dropdown(
    options={'Natural': 'natural', 'Reflected (Gray)': 'reflected'},
    value='reflected',
    description='Labeling:',
)

noise_power_db_widget = ipywidgets.FloatSlider(
    value=-40.0,
    min=-40.0,
    max=10.0,
    step=1.0,
    continuous_update=False,
    description='Noise power (dB):',
)

interactive_output = ipywidgets.interactive_output(
    psk_constellation_demo,
    dict(
        order=order_widget,
        amplitude=amplitude_widget,
        phase_offset=phase_offset_widget,
        labeling=labeling_widget,
        noise_power_db=noise_power_db_widget,
    ),
)

ipywidgets.VBox(
    [
        ipywidgets.HBox(
            [
                order_widget,
                amplitude_widget,
                phase_offset_widget,
                labeling_widget,
            ]
        ),
        noise_power_db_widget,
        interactive_output,
    ],
)


# ## Quadrature Amplitude Modulation (QAM)

# In[4]:


def qam_constellation_demo(order, base_amplitude, phase_offset, labeling, noise_power_db):
    qam_modulation = komm.QAModulation(order, base_amplitude, phase_offset, labeling)
    lim = [-2.125*np.sqrt(order), 2.125*np.sqrt(order)]
    constellation_demo(qam_modulation, noise_power_db, xlim=lim, ylim=lim)

order_widget = ipywidgets.SelectionSlider(
    options=[4, 16, 64, 256],
    continuous_update=False,
    description='Order:',
)

base_amplitude_widget = ipywidgets.FloatSlider(
    min=0.1,
    max=2.01,
    step=0.1,
    value=1.0,
    continuous_update=False,
    description='Base amplitude:',
)

phase_offset_widget = ipywidgets.SelectionSlider(
    options=[('{:.2f}π'.format(x), np.pi*x) for x in np.arange(0.0, 2.001, step=0.01)],
    value=0.0,
    continuous_update=False,
    description='Phase offset:',
)

labeling_widget = ipywidgets.Dropdown(
    options={'Natural': 'natural', 'Reflected 2D (Gray)': 'reflected_2d'},
    value='reflected_2d',
    description='Labeling:',
)


noise_power_db_widget = ipywidgets.FloatSlider(
    value=-40.0,
    min=-40.0,
    max=10.0,
    step=1.0,
    continuous_update=False,
    description='Noise power (dB):',
)

interactive_output = ipywidgets.interactive_output(
    qam_constellation_demo,
    dict(
        order=order_widget,
        base_amplitude=base_amplitude_widget,
        phase_offset=phase_offset_widget,
        labeling=labeling_widget,
        noise_power_db=noise_power_db_widget,
    ),
)

ipywidgets.VBox(
    [
        ipywidgets.HBox(
            [
                order_widget,
                base_amplitude_widget,
                phase_offset_widget,
                labeling_widget,
            ]
        ),
        noise_power_db_widget,
        interactive_output,
    ],
)

