import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# local import
from utils import show_about, show_code, show_documentation

import komm

st.set_page_config(page_title="Komm Demo: Constellations", layout="wide")


def plot_psk(order, amplitude, phase_offset, labeling):
    psk_modulation = komm.PSKModulation(order, amplitude, phase_offset, labeling)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(linestyle="--")

    ax.scatter(
        psk_modulation.constellation.real,
        psk_modulation.constellation.imag,
        s=8**2,
    )

    for i, point in enumerate(psk_modulation.constellation):
        label = "".join(str(b) for b in psk_modulation.labeling[i])
        ax.text(
            point.real,
            point.imag + 0.075 * -3.0,
            label,
            horizontalalignment="center",
        )

    ax.set_title(repr(psk_modulation))
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.axis("square")
    ax.set_xlim((-2.2, 2.2))
    ax.set_ylim((-2.2, 2.2))

    return fig


def plot_qam(order, base_amplitude, phase_offset, labeling):
    qam_modulation = komm.QAModulation(order, base_amplitude, phase_offset, labeling)
    lim = (-2.125 * np.sqrt(order), 2.125 * np.sqrt(order))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(linestyle="--")

    ax.scatter(
        qam_modulation.constellation.real,
        qam_modulation.constellation.imag,
        s=8**2,
    )

    for i, point in enumerate(qam_modulation.constellation):
        label = "".join(str(b) for b in qam_modulation.labeling[i])
        ax.text(
            point.real,
            point.imag + 0.075 * lim[0],
            label,
            horizontalalignment="center",
        )

    ax.set_title(repr(qam_modulation))
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
    show_documentation("Phase-shift keying", "PSKModulation")

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
    with col2:
        st.pyplot(plot_psk(order, amplitude, phase_offset, labeling))

    show_code(plot_psk)


with tab2:
    show_documentation("Quadrature amplitude modulation", "QAModulation")

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
            options=["natural_2d", "reflected_2d"],
            format_func=lambda x: (
                "Natural 2D" if x == "natural_2d" else "Reflected 2D (Gray)"
            ),
            index=1,
        )
    with col2:
        st.pyplot(plot_qam(order, base_amplitude, phase_offset, labeling))

    show_code(plot_qam)

show_about()
