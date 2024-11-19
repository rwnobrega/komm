import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# local import
from utils import show_about, show_documentation

import komm

st.set_page_config(page_title="Komm Demo: Pulses", layout="wide")

tab_configs = [
    {
        "name": "Rectangular",
        "doc_title": "Rectangular pulse",
        "doc_ref": "RectangularPulse",
        "pulse_class": komm.RectangularPulse,
        "t_axis": (-0.1, 1.1, -0.1, 1.1),
        "f_axis": (-15.0, 15.0, -0.3, 1.1),
        "params": [{
            "name": "width",
            "min": 0.01,
            "max": 1.0,
            "default": 1.0,
            "step": 0.01,
        }],
    },
    {
        "name": "Manchester",
        "doc_title": "Manchester pulse",
        "doc_ref": "ManchesterPulse",
        "pulse_class": komm.ManchesterPulse,
        "t_axis": (-0.1, 1.1, -1.1, 1.1),
        "f_axis": (-6.0, 6.0, -0.1, 0.6),
        "params": [],
    },
    {
        "name": "Sinc",
        "doc_title": "Sinc pulse",
        "doc_ref": "SincPulse",
        "pulse_class": komm.SincPulse,
        "t_axis": (-8.0, 8.0, -0.3, 1.1),
        "f_axis": (-1.1, 1.1, -0.1, 1.1),
        "params": [],
    },
    {
        "name": "Raised cosine",
        "doc_title": "Raised cosine pulse",
        "doc_ref": "RaisedCosinePulse",
        "pulse_class": komm.RaisedCosinePulse,
        "t_axis": (-8.0, 8.0, -0.3, 1.1),
        "f_axis": (-1.1, 1.1, -0.1, 1.1),
        "params": [{
            "name": "rolloff",
            "min": 0.0,
            "max": 1.0,
            "default": 1.0,
            "step": 0.01,
        }],
    },
    {
        "name": "Root raised cosine",
        "doc_title": "Root raised cosine pulse",
        "doc_ref": "RootRaisedCosinePulse",
        "pulse_class": komm.RootRaisedCosinePulse,
        "t_axis": (-8.0, 8.0, -0.3, 1.3),
        "f_axis": (-1.1, 1.1, -0.1, 1.1),
        "params": [{
            "name": "rolloff",
            "min": 0.0,
            "max": 1.0,
            "default": 1.0,
            "step": 0.01,
        }],
    },
    {
        "name": "Gaussian",
        "doc_title": "Gaussian pulse",
        "doc_ref": "GaussianPulse",
        "pulse_class": komm.GaussianPulse,
        "t_axis": (-1.5, 1.5, -0.1, 1.1),
        "f_axis": (-4.0, 4.0, -0.1, 0.7),
        "params": [{
            "name": "half_power_bandwidth",
            "min": 0.5,
            "max": 2.0,
            "default": 1.0,
            "step": 0.01,
        }],
    },
]

tab_names = [config["name"] for config in tab_configs]
tabs = st.tabs(tab_names)

for tab_idx, config in enumerate(tab_configs):
    with tabs[tab_idx]:
        show_documentation(config["doc_title"], config["doc_ref"])
        plot_container = st.container()
        repr_container = st.container()
        params = {}
        for param in config["params"]:
            params[param["name"]] = st.slider(
                key=f"{config['name']}_{param['name']}",
                label=param["name"],
                min_value=param["min"],
                max_value=param["max"],
                value=param["default"],
                step=param["step"],
            )
        pulse = config["pulse_class"](**params)
        t_axis = config["t_axis"]
        f_axis = config["f_axis"]

        t = np.linspace(t_axis[0], t_axis[1], 512)
        f = np.linspace(f_axis[0], f_axis[1], 512)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))

        if isinstance(pulse, komm.RectangularPulse):
            ax0.step(t, pulse.waveform(t), "b")
        else:
            ax0.plot(t, pulse.waveform(t), "b")

        ax0.axis(t_axis)
        ax0.set_title("Waveform")
        ax0.set_xlabel("$t$")
        ax0.set_ylabel("$h(t)$")
        ax0.grid()

        ax1.plot(f, pulse.spectrum(f), "r")
        ax1.axis(f_axis)
        ax1.set_title("Spectrum")
        ax1.set_xlabel("$f$")
        ax1.set_ylabel("$\\hat{h}(f)$")
        ax1.grid()

        with repr_container:
            st.write(f"komm.{pulse.__repr__()}")
        with plot_container:
            st.pyplot(fig)

show_about()
