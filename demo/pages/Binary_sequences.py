import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# local import
from utils import show_about, show_code, show_documentation

import komm

st.set_page_config(page_title="Komm Demo: Binary Sequences", layout="wide")


def plot_barker(length):
    barker = komm.BarkerSequence(length=length)
    shifts = np.arange(-2 * length + 1, 2 * length)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

    ax0.stem(np.arange(length), barker.polar_sequence)
    ax0.set_title("Sequence")
    ax0.set_xlabel("$n$")
    ax0.set_ylabel("$a[n]$")
    ax0.set_xticks(np.arange(length))
    ax0.set_yticks([-1, 0, 1])

    ax1.stem(shifts, barker.autocorrelation(shifts))
    ax1.set_title("Autocorrelation")
    ax1.set_xlabel("$\\ell$")
    ax1.set_ylabel("$R[\\ell]$")
    ax1.set_xticks([-length, 0, length])
    ax1.set_yticks(np.arange(-1, length + 1))

    return barker, fig


def plot_walsh_hadamard(length, ordering, index):
    walsh_hadamard = komm.WalshHadamardSequence(
        length=length,
        ordering=ordering,
        index=index,
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

    ax0.stem(np.arange(length), walsh_hadamard.polar_sequence)
    ax0.set_title("Sequence")
    ax0.set_xlabel("$n$")
    ax0.set_ylabel("$a[n]$")
    ax0.set_yticks([-1, 0, 1])
    ax0.set_ylim((-1.2, 1.2))

    ax1.stem(np.arange(length), walsh_hadamard.autocorrelation(normalized=True))
    ax1.set_title("Autocorrelation (normalized)")
    ax1.set_xlabel("$\\ell$")
    ax1.set_ylabel("$R[\\ell]$")

    return walsh_hadamard, fig


def plot_lfsr(degree):
    lfsr = komm.LFSRSequence.maximum_length_sequence(degree=degree)
    length = lfsr.length
    shifts = np.arange(-2 * length + 1, 2 * length)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

    ax0.stem(np.arange(length), lfsr.polar_sequence)
    ax0.set_title("Sequence")
    ax0.set_xlabel("$n$")
    ax0.set_ylabel("$a[n]$")
    ax0.set_yticks([-1, 0, 1])

    ax1.stem(shifts, lfsr.cyclic_autocorrelation(shifts, normalized=True))
    ax1.set_title("Cyclic autocorrelation (normalized)")
    ax1.set_xlabel("$\\ell$")
    ax1.set_ylabel("$\\tilde{R}[\\ell]$")
    ax1.set_xticks([-length, 0, length])
    ax1.set_ylim((-0.5, 1.1))

    return lfsr, fig


tab1, tab2, tab3 = st.tabs([
    "Barker",
    "Walsh–Hadamard",
    "Linear-feedback shift register (LFSR)",
])


with tab1:
    show_documentation("Barker sequence", "BarkerSequence")

    length = st.select_slider(
        label="Length",
        options=[2, 3, 4, 5, 7, 11, 13],
        value=2,
    )

    barker, fig = plot_barker(length)
    st.write(repr(barker))
    st.pyplot(fig)
    show_code(plot_barker)

with tab2:
    show_documentation("Walsh–Hadamard sequence", "WalshHadamardSequence")

    col1, col2, col3 = st.columns(3)
    with col1:
        length = st.select_slider(
            label="Length",
            options=[2**i for i in range(1, 8)],
            value=2,
        )
    with col2:
        ordering = st.selectbox(
            label="Ordering",
            options=["natural", "sequency"],
            index=0,
        )
    with col3:
        index = st.slider(
            label="Index",
            min_value=0,
            max_value=length - 1,
            value=0,
        )

    walsh_hadamard, fig = plot_walsh_hadamard(length, ordering, index)
    st.write(repr(walsh_hadamard))
    st.pyplot(fig)
    show_code(plot_walsh_hadamard)

with tab3:
    show_documentation("Linear-feedback shift register", "LFSRSequence")

    degree = st.slider(
        label="Degree",
        min_value=2,
        max_value=7,
        value=4,
    )

    lfsr, fig = plot_lfsr(degree)
    st.write(repr(lfsr))
    st.pyplot(fig)
    show_code(plot_lfsr)

show_about()
