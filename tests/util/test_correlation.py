import numpy as np
import pytest

from komm._util import acorr, cyclic_acorr


def test_acorr_1():
    # Wikipedia example
    assert np.allclose(acorr([2, 3, -1]), [14, 3, -2])
    assert np.allclose(acorr([2, 3, -1], normalized=True), [1, 3 / 14, -1 / 7])
    assert np.allclose(acorr([2, 3, -1], shifts=[-1, 0, 1]), [3, 14, 3])
    assert np.allclose(acorr([2, 3, -1], shifts=[-11, 2, 0, 6]), [0, -2, 14, 0])


@pytest.mark.parametrize(
    "shifts, answer",
    [
        ([0], [0.69]),
        ([-1, 0, 1], [0.22, 0.69, 0.22]),
        ([-2, -1, 0, 1, 2], [0.28, 0.22, 0.69, 0.22, 0.28]),
        ([-3, -2, -1, 0, 1, 2, 3], [0.00, 0.28, 0.22, 0.69, 0.22, 0.28, 0.00]),
        ([0, 2, 4, 6], [0.69, 0.28, 0.00, 0.00]),
    ],
)
def test_acorr_2(shifts, answer):
    seq = [0.7, 0.2, 0.4]
    assert np.allclose(acorr(seq, shifts), answer)


@pytest.mark.parametrize(
    "seq, answer",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 14.0, 26.0, 40.0, 55.0, 40.0, 26.0, 14.0, 5.0],
        ),
        ([1.0, 1.0j, -1.0, -1.0j], [1.0j, -2.0, -3.0j, 4.0, 3.0j, -2.0, -1.0j]),
    ],
)
def test_acorr_3(seq, answer):
    L = len(seq)
    assert np.allclose(np.correlate(seq, seq, mode="full"), answer)
    for max_lag in range(1, 2 * L):
        shifts = np.arange(-max_lag, max_lag + 1)
        if max_lag < L:
            expected = answer[L - max_lag - 1 : L + max_lag]
        else:
            extra = max_lag - L + 1
            expected = np.concatenate((np.zeros(extra), answer, np.zeros(extra)))
        assert np.allclose(acorr(seq, shifts), expected)


def test_cyclic_acorr_1():
    # Wikipedia example
    assert np.allclose(cyclic_acorr([2, 3, -1]), [14, 1, 1])
    assert np.allclose(cyclic_acorr([2, 3, -1], normalized=True), [1, 1 / 14, 1 / 14])
    assert np.allclose(cyclic_acorr([2, 3, -1], shifts=[-1, 0, 1]), [1, 14, 1])
    assert np.allclose(cyclic_acorr([2, 3, -1], shifts=[-11, 2, 0, 6]), [1, 1, 14, 14])


@pytest.mark.parametrize(
    "shifts, answer",
    [
        ([0], [0.69]),
        ([-1, 0, 1], [0.50, 0.69, 0.50]),
        ([-2, -1, 0, 1, 2], [0.50, 0.50, 0.69, 0.50, 0.50]),
        ([-3, -2, -1, 0, 1, 2, 3], [0.69, 0.50, 0.50, 0.69, 0.50, 0.50, 0.69]),
        ([0, 2, 4, 6], [0.69, 0.50, 0.50, 0.69]),
    ],
)
def test_cyclic_acorr_2(shifts, answer):
    seq = [0.7, 0.2, 0.4]
    assert np.allclose(cyclic_acorr(seq, shifts), answer)


@pytest.mark.parametrize(
    "seq, answer",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [45.0, 40.0, 40.0, 45.0, 55.0, 45.0, 40.0, 40.0, 45.0],
        ),
        ([1.0, 1.0j, -1.0, -1.0j], [4.0j, -4.0, -4.0j, 4.0, 4.0j, -4.0, -4.0j]),
    ],
)
def test_cyclic_acorr_3(seq, answer):
    L = len(seq)
    assert np.allclose(
        np.fft.ifft(np.fft.fft(seq) * np.conj(np.fft.fft(seq))), answer[L - 1 :]
    )
    for max_lag in range(1, 2 * L):
        shifts = np.arange(-max_lag, max_lag + 1)
        if max_lag < L:
            expected = answer[L - max_lag - 1 : L + max_lag]
        else:
            extra = max_lag - L + 1
            expected = np.concatenate(
                (answer[L - extra : L], answer, answer[L - 1 : L - 1 + extra])
            )
        assert np.allclose(cyclic_acorr(seq, shifts), expected)
