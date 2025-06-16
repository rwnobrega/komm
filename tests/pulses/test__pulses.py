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


def test_pulses_support_and_time_span():
    pulse = komm.RectangularPulse(0.25)
    assert pulse.support == (0.0, 0.25)
    assert pulse.time_span() == (0, 1)

    pulse = komm.SincPulse()
    assert pulse.support == (-np.inf, np.inf)
    assert pulse.time_span(truncation=4) == (-2, 2)


def test_pulses_taps():
    pulse = komm.RectangularPulse(0.25)
    np.testing.assert_allclose(
        pulse.taps(samples_per_symbol=3),
        [1.0, 0.0, 0.0, 0.0],
    )

    pulse = komm.SincPulse()
    np.testing.assert_allclose(
        pulse.taps(samples_per_symbol=4, truncation=4),
        # fmt: off
        [0.0, -0.128617, -0.212207, -0.180063,
         0.0, +0.300105, +0.636620, +0.900316,
         1.0, +0.900316, +0.636620, +0.300105,
         0.0, -0.180063, -0.212207, -0.128617, 0.0],
        # fmt: on
        atol=1e-6,
    )
