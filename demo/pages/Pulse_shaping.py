import matplotlib.pyplot as plt
import numpy as np
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

pulse = komm.RaisedCosinePulse(rolloff=rolloff)
info = np.array([3, 1, -1, 1, -3, -1, 1, -3, 3, 1])
x = komm.sampling_rate_expand(info, factor=sps)
p = pulse.taps(samples_per_symbol=sps, span=(-truncation // 2, truncation // 2))
y = np.convolve(x, p)
t = np.arange(y.size) / sps - truncation // 2

fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(info, linefmt="r", markerfmt="ro")
ax.plot(t, y, "o-", markersize=2)
ax.axis((-3, 12, -4, 4))
ax.set_xlabel("$t / T_b$")
ax.set_ylabel("$y(t)$")
ax.grid()

fig.tight_layout()
st.pyplot(fig)
