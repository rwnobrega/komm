import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.pylab import Axes

# local import
from utils import show_about, show_documentation

import komm
import komm.abc

st.set_page_config(page_title="Komm Demo: Pulses", layout="wide")

tab_configs = [
    {
        "name": "Rectangular",
        "doc_title": "Rectangular pulse",
        "doc_ref": "RectangularPulse",
        "pulse_class": komm.RectangularPulse,
        "ht_axis": (-0.1, 1.1, -0.1, 1.1),
        "hf_axis": (-15.0, 15.0, -0.3, 1.1),
        "R_axis": (-1.1, 1.1, -0.1, 1.1),
        "S_axis": (-1.1, 1.1, -0.1, 1.1),
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
        "ht_axis": (-0.1, 1.1, -1.1, 1.1),
        "hf_axis": (-6.0, 6.0, -0.8, 0.8),
        "R_axis": (-1.1, 1.1, -0.6, 1.1),
        "S_axis": (-6.0, 6.0, -0.05, 0.55),
        "phase_adjust": (0.5, 0.25),
        "params": [],
    },
    {
        "name": "Sinc",
        "doc_title": "Sinc pulse",
        "doc_ref": "SincPulse",
        "pulse_class": komm.SincPulse,
        "ht_axis": (-8.0, 8.0, -0.3, 1.1),
        "hf_axis": (-1.1, 1.1, -0.1, 1.1),
        "R_axis": (-8.0, 8.0, -0.3, 1.1),
        "S_axis": (-1.1, 1.1, -0.1, 1.1),
        "params": [],
    },
    {
        "name": "Raised-cosine",
        "doc_title": "Raised-cosine pulse",
        "doc_ref": "RaisedCosinePulse",
        "pulse_class": komm.RaisedCosinePulse,
        "ht_axis": (-8.0, 8.0, -0.3, 1.1),
        "hf_axis": (-1.1, 1.1, -0.1, 1.1),
        "R_axis": (-8.0, 8.0, -0.3, 1.1),
        "S_axis": (-1.1, 1.1, -0.1, 1.1),
        "params": [{
            "name": "rolloff",
            "min": 0.0,
            "max": 1.0,
            "default": 1.0,
            "step": 0.01,
        }],
    },
    {
        "name": "Root-raised-cosine",
        "doc_title": "Root-raised-cosine pulse",
        "doc_ref": "RootRaisedCosinePulse",
        "pulse_class": komm.RootRaisedCosinePulse,
        "ht_axis": (-8.0, 8.0, -0.3, 1.3),
        "hf_axis": (-1.1, 1.1, -0.1, 1.1),
        "R_axis": (-8.0, 8.0, -0.3, 1.1),
        "S_axis": (-1.1, 1.1, -0.1, 1.1),
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
        "ht_axis": (-1.5, 1.5, -0.1, 1.1),
        "hf_axis": (-4.0, 4.0, -0.05, 0.7),
        "R_axis": (-1.5, 1.5, -0.05, 0.55),
        "S_axis": (-4.0, 4.0, -0.05, 0.5),
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
        to_show = st.radio(
            label="Show:",
            options=[
                "Waveform and spectrum",
                "Autocorrelation and energy density spectrum",
            ],
            horizontal=True,
            key=tab_idx,
        )
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
        pulse: komm.abc.Pulse = config["pulse_class"](**params)

        ax: list[Axes]
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))

        if to_show == "Waveform and spectrum":
            ht_axis = config["ht_axis"]
            hf_axis = config["hf_axis"]
            t = np.linspace(ht_axis[0], ht_axis[1], 512)
            f = np.linspace(hf_axis[0], hf_axis[1], 512)
            ax[0].plot(t, pulse.waveform(t), "C0")
            ax[0].axis(ht_axis)
            ax[0].set_title("Waveform")
            ax[0].set_xlabel("$t$")
            ax[0].set_ylabel("$p(t)$")
            ax[0].grid()
            a, b = config["phase_adjust"] if "phase_adjust" in config else (0, 0)
            spectrum = pulse.spectrum(f) * np.exp(2j * np.pi * (a * f + b))
            ax[1].plot(f, np.real(spectrum), "C1")
            ax[1].axis(hf_axis)
            ax[1].set_title("Spectrum")
            ax[1].set_xlabel("$f$")
            ax[1].set_ylabel("$\\hat{p}(f)$")
            ax[1].grid()
        elif to_show == "Autocorrelation and energy density spectrum":
            R_axis = config["R_axis"]
            S_axis = config["S_axis"]
            τ = np.linspace(R_axis[0], R_axis[1], 512)
            f = np.linspace(S_axis[0], S_axis[1], 512)
            ax[0].plot(τ, pulse.autocorrelation(τ), "C0")
            ax[0].axis(R_axis)
            ax[0].set_title("Autocorrelation")
            ax[0].set_xlabel("$\\tau$")
            ax[0].set_ylabel("$R(\\tau)$")
            ax[0].grid()
            ax[1].plot(f, pulse.energy_density_spectrum(f), "C1")
            ax[1].axis(S_axis)
            ax[1].set_title("Energy density spectrum")
            ax[1].set_xlabel("$f$")
            ax[1].set_ylabel("$S(f)$")
            ax[1].grid()

        with repr_container:
            st.write(f"komm.{pulse.__repr__()}")
        with plot_container:
            st.pyplot(fig)


show_about()
