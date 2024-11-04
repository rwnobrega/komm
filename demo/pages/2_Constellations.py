import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Local import
from utils import show_about, show_code

import komm

st.set_page_config(page_title="Komm Demo: Constellations", layout="wide")


def plot_constellation(modulation, noise_power_db, xlim, ylim):
    """Base function for plotting constellation diagrams."""
    signal_power = modulation.energy_per_symbol
    noise_power = 10 ** (noise_power_db / 10)
    snr = signal_power / noise_power
    awgn = komm.AWGNChannel(signal_power=signal_power, snr=snr)

    num_symbols = 10000
    num_bits = modulation.bits_per_symbol * num_symbols
    bits = np.random.randint(2, size=num_bits)
    sentword = modulation.modulate(bits)
    recvword = awgn(sentword)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.grid(linestyle="--")

    ax.scatter(recvword.real, recvword.imag, color="xkcd:light blue", s=1)
    ax.scatter(
        modulation.constellation.real,
        modulation.constellation.imag,
        color="xkcd:blue",
        s=8**2,
    )

    for i, point in enumerate(modulation.constellation):
        label = "".join(str(b) for b in modulation.labeling[i])
        ax.text(
            point.real,
            point.imag + 0.075 * xlim[0],
            label,
            horizontalalignment="center",
        )

    ax.set_title(repr(modulation))
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.axis("square")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    info_text = f"SNR = {10*np.log10(awgn.snr):.1f} dB\n"
    info_text += f"Eb/N0 = {10*np.log10(awgn.snr / modulation.bits_per_symbol):.1f} dB"
    ax.text(
        1.125 * xlim[1],
        0.0,
        info_text,
        horizontalalignment="left",
        verticalalignment="center",
    )

    return fig


def plot_psk(order, amplitude, phase_offset, labeling, noise_power_db):
    """Plot PSK constellation."""
    psk_modulation = komm.PSKModulation(order, amplitude, phase_offset, labeling)
    return plot_constellation(
        psk_modulation, noise_power_db, xlim=[-3.0, 3.0], ylim=[-3.0, 3.0]
    )


def plot_qam(order, base_amplitude, phase_offset, labeling, noise_power_db):
    """Plot QAM constellation."""
    qam_modulation = komm.QAModulation(order, base_amplitude, phase_offset, labeling)
    lim = [-2.125 * np.sqrt(order), 2.125 * np.sqrt(order)]
    return plot_constellation(qam_modulation, noise_power_db, xlim=lim, ylim=lim)


st.sidebar.title("Constellation type")
constellation_type = st.sidebar.radio(
    "Select constellation type:",
    ["PSK", "QAM"],
    label_visibility="collapsed",
)

if constellation_type == "PSK":
    st.title("Phase-shift keying (PSK)")

    col1, col2 = st.columns([3, 5])
    with col1:
        order = st.select_slider(
            label="Order",
            options=[2, 4, 8, 16, 32],
            value=4,
        )
        amplitude = st.slider(
            label="Amplitude",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )
        phase_offset = st.slider(
            label="Phase offset (2π rad)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        ) * (2 * np.pi)
        labeling = st.selectbox(
            label="Labeling",
            options=["natural", "reflected"],
            format_func=lambda x: "Natural" if x == "natural" else "Reflected (Gray)",
            index=1,
        )
        noise_power_db = st.slider(
            label="Noise power (dB)",
            min_value=-40,
            max_value=10,
            value=-40,
            step=1,
        )

    with col2:
        st.pyplot(plot_psk(order, amplitude, phase_offset, labeling, noise_power_db))

    with st.expander("Source code"):
        st.code(show_code(plot_psk), language="python")

else:  # QAM
    st.title("Quadrature amplitude modulation (QAM)")

    col1, col2 = st.columns([3, 5])
    with col1:
        order = st.select_slider(
            label="Order",
            options=[4, 16, 64, 256],
            value=16,
        )
        base_amplitude = st.slider(
            label="Base amplitude",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )
        phase_offset = st.slider(
            label="Phase offset (2π rad)",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.01,
        ) * (2 * np.pi)
        labeling = st.selectbox(
            label="Labeling",
            options=["natural", "reflected_2d"],
            format_func=lambda x: (
                "Natural" if x == "natural" else "Reflected 2D (Gray)"
            ),
            index=1,
        )
        noise_power_db = st.slider(
            label="Noise power (dB)",
            min_value=-40.0,
            max_value=10.0,
            value=-40.0,
            step=1.0,
        )

    with col2:
        st.pyplot(
            plot_qam(order, base_amplitude, phase_offset, labeling, noise_power_db)
        )

    with st.expander("Source code"):
        st.code(show_code(plot_qam), language="python")

st.sidebar.divider()
show_about()
