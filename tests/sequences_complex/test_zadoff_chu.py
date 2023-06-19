import itertools as it

import numpy as np
import pytest

import komm


def test_zadoff_chu_1():
    # Andrews.22, Sec. III, Example 1
    s_1 = komm.ZadoffChuSequence(5, root_index=1).sequence
    assert np.allclose(
        s_1,
        np.array([1.0, np.exp(-2j * np.pi / 5), np.exp(-6j * np.pi / 5), np.exp(-2j * np.pi / 5), 1.0]),
    )
    assert np.allclose(
        np.roll(s_1, -2),
        np.array([np.exp(-6j * np.pi / 5), np.exp(-2j * np.pi / 5), 1.0, 1.0, np.exp(-2j * np.pi / 5)]),
    )
    assert np.allclose(np.vdot(s_1, np.roll(s_1, -2)), 0.0)
    s_4 = komm.ZadoffChuSequence(5, root_index=4).sequence
    assert np.allclose(
        s_4,
        np.array([1.0, np.exp(2j * np.pi / 5), np.exp(-4j * np.pi / 5), np.exp(2j * np.pi / 5), 1.0]),
    )
    for shift in range(5):
        assert np.allclose(np.abs(np.vdot(s_1, np.roll(s_4, shift))), np.sqrt(5))


@pytest.mark.parametrize("length", range(1, 40, 2))
def test_zadoff_chu_2(length):
    # Test for constant amplitude
    for q in range(1, length):
        zc = komm.ZadoffChuSequence(length, root_index=q)
        assert np.allclose(np.abs(zc.sequence), 1.0)


@pytest.mark.parametrize("length", range(1, 40, 2))
def test_zadoff_chu_3(length):
    # Test for zero cyclic autocorrelation
    for q in range(1, length):
        if np.gcd(q, length) != 1:
            continue
        zc = komm.ZadoffChuSequence(length, root_index=q)
        expected_acorr = np.zeros(length)
        expected_acorr[0] = 1.0
        assert np.allclose(
            zc.cyclic_autocorrelation(normalized=True),
            expected_acorr,
        )


@pytest.mark.parametrize("length", range(1, 20, 2))
def test_zadoff_chu_4(length):
    # Test for constant cyclic cross-correlation
    for q1, q2 in it.combinations(range(1, length), 2):
        if np.gcd(abs(q1 - q2), length) != 1:
            continue
        zc1 = komm.ZadoffChuSequence(length, root_index=q1)
        zc2 = komm.ZadoffChuSequence(length, root_index=q2)
        for shift in range(length):
            assert np.allclose(
                np.abs(np.vdot(zc1.sequence, np.roll(zc2.sequence, shift))),
                np.sqrt(length),
            )
