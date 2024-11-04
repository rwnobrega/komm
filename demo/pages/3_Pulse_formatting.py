import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Local import
from utils import show_about, show_code

import komm

st.set_page_config(page_title="Komm Demo: Pulse Formatting", layout="wide")


def plot_sinc():
    pulse = komm.SincPulse(length_in_symbols=20)
    h = pulse.impulse_response
    H = pulse.frequency_response
    t = np.linspace(-8.0, 8.0, 1000)
    f = np.linspace(-1.5, 1.5, 200)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    ax0.plot(t, h(t), "b")
    ax0.axis([-7.1, 7.1, -0.3, 1.1])
    ax0.set_title("Sinc pulse (waveform)")
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("$h(t)$")
    ax0.grid()

    ax1.plot(f, H(f), "r")
    ax1.axis([-1.1, 1.1, -0.1, 1.1])
    ax1.set_title("Sinc pulse (spectrum)")
    ax1.set_xlabel("$f$")
    ax1.set_ylabel("$H(f)$")
    ax1.grid()

    return fig


def plot_raised_cosine(rolloff):
    pulse = komm.RaisedCosinePulse(rolloff, length_in_symbols=16)
    h = pulse.impulse_response
    H = pulse.frequency_response
    t = np.linspace(-8.0, 8.0, 1000)
    f = np.linspace(-1.5, 1.5, 200)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    ax0.plot(t, h(t), "b")
    ax0.axis([-7.1, 7.1, -0.3, 1.1])
    ax0.set_title("Raised cosine pulse (waveform)")
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("$h(t)$")
    ax0.grid()

    ax1.plot(f, H(f), "r")
    ax1.axis([-1.1, 1.1, -0.1, 1.1])
    ax1.set_title("Raised cosine pulse (spectrum)")
    ax1.set_xlabel("$f$")
    ax1.set_ylabel("$H(f)$")
    ax1.grid()

    return fig


def plot_gaussian(half_power_bandwidth):
    pulse = komm.GaussianPulse(half_power_bandwidth, length_in_symbols=4)
    h = pulse.impulse_response
    H = pulse.frequency_response
    t = np.linspace(-8.0, 8.0, 1000)
    f = np.linspace(-4.0, 4.0, 500)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    ax0.plot(t, h(t), "b")
    ax0.axis([-7.1, 7.1, -0.1, 1.1])
    ax0.set_title("Gaussian pulse (waveform)")
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("$h(t)$")
    ax0.grid()

    ax1.plot(f, H(f), "r")
    ax1.plot(
        [-4.0, 4.0],
        [H(0) / np.sqrt(2), H(0) / np.sqrt(2)],
        linestyle="dashed",
        color="gray",
    )
    ax1.plot(
        [half_power_bandwidth, half_power_bandwidth],
        [-0.1 * H(0), 1.1 * H(0)],
        linestyle="dashed",
        color="gray",
    )
    ax1.plot(
        [-half_power_bandwidth, -half_power_bandwidth],
        [-0.1 * H(0), 1.1 * H(0)],
        linestyle="dashed",
        color="gray",
    )
    ax1.axis([-2.0, 2.0, -0.1 * H(0), 1.1 * H(0)])
    ax1.set_title("Gaussian pulse (spectrum)")
    ax1.set_xlabel("$f$")
    ax1.set_ylabel("$H(f)$")
    ax1.grid()

    return fig


st.sidebar.title("Pulse type")
pulse_type = st.sidebar.radio(
    "Select pulse type:",
    ["Sinc", "Raised cosine", "Gaussian"],
    label_visibility="collapsed",
)

if pulse_type == "Sinc":
    st.title("Sinc pulse")

    st.pyplot(plot_sinc())

    with st.expander("Source code"):
        st.code(show_code(plot_sinc), language="python")

elif pulse_type == "Raised cosine":
    st.title("Raised cosine pulse")

    rolloff = st.slider(
        label="Roll-off factor",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    st.pyplot(plot_raised_cosine(rolloff))

    with st.expander("Source code"):
        st.code(show_code(plot_raised_cosine), language="python")

else:  # Gaussian
    st.title("Gaussian pulse")

    half_power_bandwidth = st.slider(
        label="Half-power bandwidth",
        min_value=0.05,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    st.pyplot(plot_gaussian(half_power_bandwidth))

    with st.expander("Source code"):
        st.code(show_code(plot_gaussian), language="python")

st.sidebar.divider()
show_about()
