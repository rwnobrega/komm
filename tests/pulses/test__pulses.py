import numpy as np
import pytest

import komm
import komm.abc

PULSES = [
    komm.RectangularPulse(),
    komm.ManchesterPulse(),
    # komm.SincPulse(),
    komm.RaisedCosinePulse(0.3),
    komm.RaisedCosinePulse(0.75),
    komm.RaisedCosinePulse(),
    komm.RootRaisedCosinePulse(0.25),
    komm.RootRaisedCosinePulse(0.5),
    komm.RootRaisedCosinePulse(),
    komm.GaussianPulse(0.3),
    komm.GaussianPulse(0.75),
    komm.GaussianPulse(),
]


@pytest.mark.parametrize("pulse", PULSES + [komm.SincPulse()])
def test_pulses_spectrum_vs_energy_density_spectrum(pulse: komm.abc.Pulse):
    fs = np.linspace(-2.0, 2.0, 101)
    S = np.abs(pulse.spectrum(fs)) ** 2
    np.testing.assert_allclose(S, pulse.energy_density_spectrum(fs))


@pytest.mark.parametrize("pulse", PULSES)
def test_pulses_energy_density_spectrum_vs_autocorrelation(pulse: komm.abc.Pulse):
    τs = np.linspace(-20.0, 20.0, 10001)
    fs = np.linspace(-2.0, 2.0, 101)
    R = pulse.autocorrelation(τs)
    S = np.array([np.trapezoid(R * np.cos(-2 * np.pi * f * τs), τs) for f in fs])
    np.testing.assert_allclose(S, pulse.energy_density_spectrum(fs), atol=1e-3)


@pytest.mark.parametrize("pulse", PULSES)
def test_pulses_waveform_vs_autocorrelation(pulse: komm.abc.Pulse):
    ts = np.linspace(-20.0, 20.0, 10001)
    τs = np.linspace(-2.0, 2.0, 101)
    h = pulse.waveform(ts)
    R = np.array([np.trapezoid(h * np.interp(ts + τ, ts, h), ts) for τ in τs])
    np.testing.assert_allclose(R, pulse.autocorrelation(τs), atol=1e-3)
