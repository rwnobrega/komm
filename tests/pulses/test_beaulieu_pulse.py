import numpy as np
import pytest

import komm


def test_beaulieu_pulse_repr():
    pulse = komm.BeaulieuPulse(rolloff=0.25)
    assert repr(pulse) == "BeaulieuPulse(rolloff=0.25)"
    pulse = komm.BeaulieuPulse(rolloff=0.25).root()
    assert repr(pulse) == "BeaulieuPulse(rolloff=0.25).root()"


@pytest.mark.parametrize("rolloff", [-0.1, 1.5, 2.0])
def test_beaulieu_pulse_invalid_rolloff(rolloff):
    with pytest.raises(ValueError, match="must satisfy 0 <= rolloff <= 1"):
        komm.BeaulieuPulse(rolloff=rolloff)
    with pytest.raises(ValueError, match="must satisfy 0 <= rolloff <= 1"):
        komm.BeaulieuPulse(rolloff=rolloff).root()


@pytest.mark.parametrize("rolloff", [0.25, 0.35, 0.5, 0.75, 1.0])
def test_beaulieu_pulse_waveform_vs_spectrum(rolloff):
    # The waveform must be the inverse Fourier transform of the spectrum. Since
    # the spectrum is zero outside [-f_2, f_2], the integral below has no
    # truncation error. This also covers the autocorrelation of the square-root
    # version, which equals the waveform of the base pulse.
    pulse = komm.BeaulieuPulse(rolloff)
    f2 = (1 + rolloff) / 2
    fs = np.linspace(-f2, f2, 100001)
    spectrum = np.real(pulse.spectrum(fs))
    τs = np.array([-15.0, -7.3, -2.5, -0.5, 0.0, 0.5, 2.5, 7.3, 15.0])
    expected = [np.trapezoid(spectrum * np.cos(2 * np.pi * fs * τ), fs) for τ in τs]
    np.testing.assert_allclose(pulse.waveform(τs), expected, atol=1e-6)


@pytest.mark.parametrize("rolloff", [0.25, 0.35, 0.5, 0.75, 1.0])
def test_beaulieu_pulse_autocorrelation_vs_energy_spectral_density(rolloff):
    # The autocorrelation must be the inverse Fourier transform of the energy
    # spectral density. Since the latter is zero outside [-f_2, f_2], the
    # integral below has no truncation error. (This identity is not covered by
    # the corresponding shared test in test__pulses.py, whose time window is
    # too short for the 1/τ² decay of the autocorrelation of this pulse.)
    pulse = komm.BeaulieuPulse(rolloff)
    f2 = (1 + rolloff) / 2
    fs = np.linspace(-f2, f2, 100001)
    esd = pulse.energy_spectral_density(fs)
    τs = np.array([-15.0, -7.3, -2.5, -0.5, 0.0, 0.5, 2.5, 7.3, 15.0])
    expected = [np.trapezoid(esd * np.cos(2 * np.pi * fs * τ), fs) for τ in τs]
    np.testing.assert_allclose(pulse.autocorrelation(τs), expected, atol=1e-6)


@pytest.mark.parametrize("rolloff", [0.25, 0.35, 0.5, 0.75, 1.0])
def test_root_beaulieu_pulse_waveform_vs_spectrum(rolloff):
    # The waveform of the square-root version, which is computed numerically,
    # must be the inverse Fourier transform of the square root of the spectrum
    # of the base pulse. The reference below is a dense trapezoidal rule with
    # no truncation error.
    pulse = komm.BeaulieuPulse(rolloff).root()
    f2 = (1 + rolloff) / 2
    fs = np.linspace(-f2, f2, 200001)
    root_spectrum = np.sqrt(np.real(komm.BeaulieuPulse(rolloff).spectrum(fs)))
    τs = np.array([-15.0, -7.3, -2.5, -0.5, 0.0, 0.5, 2.5, 7.3, 15.0])
    expected = [
        np.trapezoid(root_spectrum * np.cos(2 * np.pi * fs * τ), fs) for τ in τs
    ]
    np.testing.assert_allclose(pulse.waveform(τs), expected, atol=1e-6)


@pytest.mark.parametrize("rolloff", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_beaulieu_pulse_is_nyquist(rolloff):
    # A Nyquist pulse must satisfy p(n) = δ[n] at the integers.
    pulse = komm.BeaulieuPulse(rolloff)
    ns = np.arange(-8, 9)
    np.testing.assert_allclose(pulse.waveform(ns), ns == 0, atol=1e-12)


def test_beaulieu_pulse_reduces_to_sinc():
    beaulieu = komm.BeaulieuPulse(rolloff=0.0)
    sinc = komm.SincPulse()
    τs = np.linspace(-8.0, 8.0, 321)
    fs = np.linspace(-1.5, 1.5, 301)
    np.testing.assert_allclose(beaulieu.waveform(τs), sinc.waveform(τs))
    np.testing.assert_allclose(beaulieu.spectrum(fs), sinc.spectrum(fs))
    np.testing.assert_allclose(beaulieu.autocorrelation(τs), sinc.autocorrelation(τs))
    np.testing.assert_allclose(beaulieu.root().waveform(τs), sinc.waveform(τs))
    assert beaulieu.energy() == sinc.energy()


@pytest.mark.parametrize("timing_error", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("rolloff", [0.25, 0.35, 0.5, 0.75, 1.0])
def test_beaulieu_pulse_is_better_than_raised_cosine(rolloff, timing_error):
    # The defining claim of [BTD01]: in the presence of a symbol timing error ε,
    # the maximum distortion D(ε) = Σ_{n ≠ 0} |p(n + ε)| of the Beaulieu pulse
    # is smaller than that of the raised-cosine pulse with the same roll-off
    # factor.
    ns = np.concatenate([np.arange(-2000, 0), np.arange(1, 2001)])
    beaulieu = np.sum(np.abs(komm.BeaulieuPulse(rolloff).waveform(ns + timing_error)))
    rc = np.sum(np.abs(komm.RaisedCosinePulse(rolloff).waveform(ns + timing_error)))
    assert beaulieu < rc
