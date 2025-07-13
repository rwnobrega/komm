import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# local import
from utils import show_about, show_code, show_documentation

import komm

st.set_page_config(page_title="Komm Demo: Constellations", layout="wide")


def plot_psk(order, amplitude, phase_offset):
    constellation = komm.PSKConstellation(order, amplitude, phase_offset)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(linestyle="--")
    ax.scatter(constellation.matrix.real, constellation.matrix.imag, s=8**2)
    for i, point in enumerate(constellation.matrix):
        x, y = point[0].real, point[0].imag
        ax.text(x, y + 0.075 * -3.0, f"$s_{{{i}}}$", horizontalalignment="center")
    ax.set_title(repr(constellation))
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.axis("square")
    ax.set_xlim((-2.2, 2.2))
    ax.set_ylim((-2.2, 2.2))
    return fig


def plot_qam(order, delta, phase_offset):
    constellation = komm.QAMConstellation(order, delta, phase_offset)
    lim = (-2.125 * np.sqrt(order), 2.125 * np.sqrt(order))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(linestyle="--")
    ax.scatter(constellation.matrix.real, constellation.matrix.imag, s=8**2)
    for i, point in enumerate(constellation.matrix):
        x, y = point[0].real, point[0].imag
        ax.text(x, y + 0.075 * lim[0], f"$s_{{{i}}}$", horizontalalignment="center")
    ax.set_title(repr(constellation))
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.axis("square")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    return fig


tab1, tab2 = st.tabs(
    ["Phase-shift keying (PSK)", "Quadrature amplitude modulation (QAM)"]
)

with tab1:
    show_documentation("Phase-shift keying", "PSKConstellation")

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
            label="Phase offset (turns)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key="psk_phase_offset",
        )
    with col2:
        st.pyplot(plot_psk(order, amplitude, phase_offset))

    show_code(plot_psk)


with tab2:
    show_documentation("Quadrature amplitude modulation", "QAMConstellation")

    col1, col2 = st.columns([3, 5])
    with col1:
        order = st.select_slider(
            label="Order",
            options=[4, 16, 64, 256],
            value=16,
        )
        delta = st.slider(
            label="Distance",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.2,
        )
        phase_offset = st.slider(
            label="Phase offset (turns)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key="qam_phase_offset",
        )
    with col2:
        st.pyplot(plot_qam(order, delta, phase_offset))

    show_code(plot_qam)

show_about()
