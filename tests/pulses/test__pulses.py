import numpy as np
import pytest

import komm
import komm.abc

PULSES = [
    komm.RectangularPulse(),
    komm.ManchesterPulse(),
    # komm.SincPulse(),
    komm.RaisedCosinePulse(rolloff=0.25),
    komm.RaisedCosinePulse(rolloff=0.3),
    komm.RaisedCosinePulse(rolloff=0.75),
    komm.RaisedCosinePulse(rolloff=1.0),
    komm.RaisedCosinePulse(),
    komm.RaisedCosinePulse(rolloff=0.25).root(),
    komm.RaisedCosinePulse(rolloff=0.5).root(),
    komm.RaisedCosinePulse(rolloff=1.0).root(),
    komm.RaisedCosinePulse().root(),
    komm.GaussianPulse(0.3),
    komm.GaussianPulse(0.75),
    komm.GaussianPulse(),
]


@pytest.mark.parametrize("pulse", PULSES + [komm.SincPulse()], ids=repr)
def test_pulses_spectrum_vs_energy_spectral_density(pulse: komm.abc.Pulse):
    fs = np.linspace(-2.0, 2.0, 101)
    S = np.abs(pulse.spectrum(fs)) ** 2
    np.testing.assert_allclose(S, pulse.energy_spectral_density(fs))


@pytest.mark.parametrize("pulse", PULSES + [komm.SincPulse()], ids=repr)
def test_pulses_energy_vs_autocorrelation(pulse: komm.abc.Pulse):
    np.testing.assert_allclose([pulse.energy()], pulse.autocorrelation([0.0]))


@pytest.mark.parametrize("pulse", PULSES, ids=repr)
def test_pulses_energy_spectral_density_vs_autocorrelation(pulse: komm.abc.Pulse):
    τs = np.linspace(-20.0, 20.0, 10001)
    fs = np.linspace(-2.0, 2.0, 101)
    R = pulse.autocorrelation(τs)
    S = np.array([np.trapezoid(R * np.cos(-2 * np.pi * f * τs), τs) for f in fs])
    np.testing.assert_allclose(S, pulse.energy_spectral_density(fs), atol=1e-3)


@pytest.mark.parametrize("pulse", PULSES, ids=repr)
def test_pulses_waveform_vs_autocorrelation(pulse: komm.abc.Pulse):
    ts = np.linspace(-20.0, 20.0, 10001)
    τs = np.linspace(-2.0, 2.0, 101)
    h = pulse.waveform(ts)
    R = np.array([np.trapezoid(h * np.interp(ts + τ, ts, h), ts) for τ in τs])
    np.testing.assert_allclose(R, pulse.autocorrelation(τs), atol=1e-3)


def test_pulses_taps():
    pulse = komm.RectangularPulse(0.25)
    np.testing.assert_allclose(
        pulse.taps(samples_per_symbol=3),
        [1.0, 0.0, 0.0, 0.0],
    )

    pulse = komm.SincPulse()
    np.testing.assert_allclose(
        pulse.taps(samples_per_symbol=4, span=(-2, 2)),
        # fmt: off
        [0.0, -0.128617, -0.212207, -0.180063,
         0.0, +0.300105, +0.636620, +0.900316,
         1.0, +0.900316, +0.636620, +0.300105,
         0.0, -0.180063, -0.212207, -0.128617, 0.0],
        # fmt: on
        atol=1e-6,
    )


@pytest.mark.parametrize("pulse", PULSES, ids=repr)
@pytest.mark.parametrize("method", ["waveform", "autocorrelation"])
def test_pulses_scalar_input_gives_scalar_output(pulse, method):
    result = getattr(pulse, method)(0.5)
    assert np.ndim(result) == 0


@pytest.mark.parametrize("pulse", PULSES, ids=repr)
@pytest.mark.parametrize("method", ["waveform", "autocorrelation"])
def test_pulses_array_input_preserves_shape(pulse, method):
    t = np.linspace(-2.0, 2.0, 7)
    result = getattr(pulse, method)(t)
    assert result.shape == t.shape
    result_2d = getattr(pulse, method)(t.reshape(1, -1))
    assert result_2d.shape == (1, t.size)


@pytest.mark.parametrize("pulse", PULSES, ids=repr)
def test_pulses_scalar_and_array_values_agree(pulse):
    ts = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for method in ["waveform", "autocorrelation"]:
        array_result = getattr(pulse, method)(ts)
        for t, expected in zip(ts, array_result):
            scalar_result = getattr(pulse, method)(t)
            np.testing.assert_allclose(np.asarray(scalar_result), expected)


def test_sinc_pulse_root():
    assert komm.SincPulse().root() == komm.SincPulse()


@pytest.mark.parametrize(
    "pulse",
    [
        komm.RectangularPulse(),
        komm.ManchesterPulse(),
        komm.GaussianPulse(),
        komm.RaisedCosinePulse(rolloff=0.25).root(),
    ],
    ids=repr,
)
def test_pulses_root_not_implemented(pulse: komm.abc.Pulse):
    with pytest.raises(NotImplementedError, match="Nyquist pulses"):
        pulse.root()
