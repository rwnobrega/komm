import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# local import
from utils import show_about, show_code, show_documentation

import komm

st.set_page_config(page_title="Komm Demo: Constellations", layout="wide")


def plot_constellation(constellation, lim, label_max):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(linestyle="--")
    ax.scatter(constellation.matrix.real, constellation.matrix.imag, s=8**2)
    if constellation.order <= label_max:
        for i, point in enumerate(constellation.matrix):
            x, y = point[0].real, point[0].imag
            ax.text(x, y + 0.075 * lim[0], f"$x_{{{i}}}$", ha="center")
    ax.axis("square")
    ax.set(title=repr(constellation), xlabel="Re", ylabel="Im", xlim=lim, ylim=lim)
    return fig


def plot_psk(order, amplitude, phase_offset):
    constellation = komm.PSKConstellation(order, amplitude, phase_offset)
    return plot_constellation(constellation, (-2.2, 2.2), label_max=16)


def plot_qam(order, delta, phase_offset):
    constellation = komm.QAMConstellation(order, delta, phase_offset)
    lim = 2.125 * np.sqrt(order)
    return plot_constellation(constellation, (-lim, lim), label_max=64)


def plot_cross_qam(order, delta, phase_offset):
    constellation = komm.CrossQAMConstellation(order, delta, phase_offset)
    lim = 2.125 * np.sqrt(order)
    return plot_constellation(constellation, (-lim, lim), label_max=32)


def constellation_tab(doc_title, doc_class, plot_fn, orders, default_order, param, key):
    show_documentation(doc_title, doc_class)
    cols = st.columns([3, 5])
    with cols[0]:
        order = st.segmented_control(
            label="Order $M$:",
            options=orders,
            default=default_order,
            required=True,
            key=f"{key}_order",
        )
        param_value = st.slider(key=f"{key}_param", **param)
        phase_offset = st.slider(
            label="Phase offset $\\phi$ (turns):",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key=f"{key}_phase",
        )
    with cols[1]:
        st.pyplot(plot_fn(order, param_value, phase_offset))
    show_code(plot_fn)


tab_psk, tab_qam, tab_cross = st.tabs([
    "Phase-shift keying (PSK)",
    "Quadrature amplitude modulation (QAM)",
    "Cross QAM constellation",
])

with tab_psk:
    constellation_tab(
        doc_title="Phase-shift keying",
        doc_class="PSKConstellation",
        plot_fn=plot_psk,
        orders=[2, 4, 8, 16, 32],
        default_order=4,
        param=dict(
            label="Amplitude $A$:",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
        ),
        key="psk",
    )

with tab_qam:
    constellation_tab(
        doc_title="Quadrature amplitude modulation",
        doc_class="QAMConstellation",
        plot_fn=plot_qam,
        orders=[4, 16, 64, 256],
        default_order=16,
        param=dict(
            label="Distance $\\delta$:",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.2,
        ),
        key="qam",
    )

with tab_cross:
    constellation_tab(
        doc_title="Cross QAM constellation",
        doc_class="CrossQAMConstellation",
        plot_fn=plot_cross_qam,
        orders=[32, 128, 512],
        default_order=32,
        param=dict(
            label="Distance $\\delta$:",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.2,
        ),
        key="cross_qam",
    )

show_about()
