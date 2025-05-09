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
    sps = st.slider(
        "Samples per symbol",
        min_value=2,
        max_value=32,
        value=16,
        step=2,
    )
with cols[1]:
    rolloff = st.slider(
        "Roll-off factor",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
    )
with cols[2]:
    truncation = st.slider(
        "Truncation",
        min_value=2,
        max_value=16,
        value=8,
        step=2,
    )

dms = komm.DiscreteMemorylessSource(
    np.ones(4) / 4,
    rng=np.random.default_rng(seed=42),
)

bits = dms(10)
seq = 2 * bits - 3

pulse = komm.RaisedCosinePulse(rolloff=rolloff)
tx_filter = komm.TransmitFilter(
    pulse,
    samples_per_symbol=sps,
    truncation=truncation,
)
t, _ = tx_filter.axes(seq)
yt = tx_filter(seq)

fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(seq, linefmt="r", markerfmt="ro")
ax.plot(t, yt, "o-", markersize=2)
ax.axis((-3, 12, -4, 4))
ax.set_xlabel("$t / T_b$")
ax.set_ylabel("$y(t)$")
ax.grid()

fig.tight_layout()
st.pyplot(fig)
