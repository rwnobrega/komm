import numpy as np
import pytest

import komm


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_gaussian_channel_complex_dtypes_get_complex_noise(dtype):
    channel = komm.GaussianChannel(noise_power=1.0)
    input = np.zeros(100_000, dtype=dtype)
    output = channel.transmit(input)
    assert np.iscomplexobj(output)
    assert np.any(output.imag != 0.0)
    np.testing.assert_allclose(np.var(output.real), 0.5, rtol=0.05)
    np.testing.assert_allclose(np.var(output.imag), 0.5, rtol=0.05)
    np.testing.assert_allclose(np.var(output), 1.0, rtol=0.05)


def test_gaussian_channel_real_input_noise_power():
    channel = komm.GaussianChannel(noise_power=1.0)
    output = channel.transmit(np.zeros(100_000))
    assert not np.iscomplexobj(output)
    np.testing.assert_allclose(np.var(output), 1.0, rtol=0.05)


def test_gaussian_channel_invalid_noise_power():
    with pytest.raises(ValueError, match="'noise_power' must be a non-negative real"):
        komm.GaussianChannel(noise_power=-1.0)
    with pytest.raises(TypeError, match="'noise_power' must be a real number"):
        komm.GaussianChannel(noise_power="1.0")  # type: ignore
    assert type(komm.GaussianChannel(noise_power=1).noise_power) is float
