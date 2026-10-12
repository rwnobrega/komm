from re import escape

import numpy as np
import pytest

import komm


def test_fourier_transform_2d_rows_match_1d():
    waveform = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    spectrum, frequencies = komm.fourier_transform(waveform, time_step=0.1)
    for i in range(waveform.shape[0]):
        spectrum_i, frequencies_i = komm.fourier_transform(waveform[i], time_step=0.1)
        np.testing.assert_allclose(spectrum[i], spectrum_i)
        np.testing.assert_allclose(frequencies, frequencies_i)


def test_fourier_transform_axis_0():
    waveform = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]).T
    spectrum, _ = komm.fourier_transform(waveform, time_step=0.1, axis=0)
    for i in range(waveform.shape[1]):
        spectrum_i, _ = komm.fourier_transform(waveform[:, i], time_step=0.1)
        np.testing.assert_allclose(spectrum[:, i], spectrum_i)


def test_fourier_transform_3d(rng):
    waveform = rng.normal(size=(2, 3, 8))
    spectrum, _ = komm.fourier_transform(waveform, time_step=0.5)
    for i in range(2):
        for j in range(3):
            spectrum_ij, _ = komm.fourier_transform(waveform[i, j], time_step=0.5)
            np.testing.assert_allclose(spectrum[i, j], spectrum_ij)


@pytest.mark.parametrize("time_step", [0.0, -0.1])
def test_fourier_transform_invalid_time_step(time_step):
    message = f"'time_step' must be a positive real number (got {time_step})"
    with pytest.raises(ValueError, match=escape(message)):
        komm.fourier_transform([1, 2, 3, 4], time_step=time_step)


def test_fourier_transform_time_step_not_real():
    with pytest.raises(TypeError, match="'time_step' must be a real number"):
        komm.fourier_transform([1, 2, 3, 4], time_step="0.1")  # type: ignore


@pytest.mark.parametrize("nfft", [0, -1])
def test_fourier_transform_invalid_nfft(nfft):
    message = f"'nfft' must be a positive integer (got {nfft})"
    with pytest.raises(ValueError, match=escape(message)):
        komm.fourier_transform([1, 2, 3, 4], time_step=0.1, nfft=nfft)


def test_fourier_transform_nfft_not_integer():
    with pytest.raises(TypeError, match=r"'nfft' must be an integer \(got float\)"):
        komm.fourier_transform([1, 2, 3, 4], time_step=0.1, nfft=8.0)  # type: ignore
