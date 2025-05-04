import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import streamlit as st

import komm

st.set_page_config(page_title="Raised Cosine Pulse", layout="wide")

st.title("Pulse shaping")
st.header("Raised cosine pulse")

cols = st.columns(3)
with cols[0]:
    rolloff = st.slider(
        "Roll-off factor",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
    )
with cols[1]:
    samples_per_symbol = st.slider(
        "Samples per symbol",
        min_value=2,
        max_value=32,
        value=16,
        step=2,
    )
with cols[2]:
    truncation = st.slider(
        "Truncation",
        min_value=2,
        max_value=16,
        value=8,
        step=2,
    )

seq = [1, -1, 1, 1, -1, -1, 1]

pulse = komm.RaisedCosinePulse(rolloff=rolloff)
tx_filter = komm.TransmitFilter(
    pulse,
    samples_per_symbol=samples_per_symbol,
    truncation=truncation,
)
t = tx_filter.time(seq)
waveform = tx_filter(seq)
spectrum = fft.fftshift(fft.fft(waveform, 1024)) / samples_per_symbol

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
ax0.stem(seq, linefmt="r", markerfmt="ro")
ax0.plot(t, waveform, "o-", markersize=2)
ax0.axis((-3, 9, -2, 2))
ax0.set_xlabel("$t$")
ax0.set_ylabel("$y(t)$")
ax0.grid()

f = np.linspace(-0.5, 0.5, 1024, endpoint=False)
ax1.plot(f, np.abs(spectrum))
ax1.axis((-0.5, 0.5, -0.5, 4.5))
ax1.set_xlabel("$f$")
ax1.set_ylabel("$|\\hat{y}(f)|$")
ax1.grid()


st.pyplot(fig)
