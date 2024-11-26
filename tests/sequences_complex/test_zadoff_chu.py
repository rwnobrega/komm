import itertools as it

import numpy as np
import pytest

import komm


def test_zadoff_chu_andrews():
    # Andrews.22, Sec. III, Example 1
    s_1 = komm.ZadoffChuSequence(5, root_index=1).sequence
    np.testing.assert_array_almost_equal(
        s_1,
        # fmt: off
        [1.0, np.exp(-2j * np.pi / 5), np.exp(-6j * np.pi / 5), np.exp(-2j * np.pi / 5), 1.0],
        # fmt: on
    )
    np.testing.assert_array_almost_equal(
        np.roll(s_1, -2),
        # fmt: off
        [np.exp(-6j * np.pi / 5), np.exp(-2j * np.pi / 5), 1.0, 1.0, np.exp(-2j * np.pi / 5)],
        # fmt: on
    )
    np.testing.assert_array_almost_equal(
        np.vdot(s_1, np.roll(s_1, -2)),
        0.0,
    )

    s_4 = komm.ZadoffChuSequence(5, root_index=4).sequence
    np.testing.assert_array_almost_equal(
        s_4,
        # fmt: off
        [1.0, np.exp(2j * np.pi / 5), np.exp(-4j * np.pi / 5), np.exp(2j * np.pi / 5), 1.0],
        # fmt: on
    )

    for shift in range(5):
        np.testing.assert_array_almost_equal(
            np.abs(np.vdot(s_1, np.roll(s_4, shift))),
            np.sqrt(5),
        )


@pytest.mark.parametrize("length", range(1, 40, 2))
def test_zadoff_chu_constant_amplitude(length):
    for q in range(1, length):
        zc = komm.ZadoffChuSequence(length, root_index=q)
        np.testing.assert_array_almost_equal(np.abs(zc.sequence), 1.0)


@pytest.mark.parametrize("length", range(1, 40, 2))
def test_zadoff_chu_zero_cyclic_acorr(length):
    for q in range(1, length):
        if np.gcd(q, length) != 1:
            continue
        zc = komm.ZadoffChuSequence(length, root_index=q)
        expected_acorr = np.zeros(length)
        expected_acorr[0] = 1.0
        np.testing.assert_array_almost_equal(
            zc.cyclic_autocorrelation(normalized=True),
            expected_acorr,
        )


@pytest.mark.parametrize("length", range(1, 20, 2))
def test_zadoff_chu_constant_cyclic_xcorr(length):
    for q1, q2 in it.combinations(range(1, length), 2):
        if np.gcd(abs(q1 - q2), length) != 1:
            continue
        zc1 = komm.ZadoffChuSequence(length, root_index=q1)
        zc2 = komm.ZadoffChuSequence(length, root_index=q2)
        for shift in range(length):
            np.testing.assert_array_almost_equal(
                np.abs(np.vdot(zc1.sequence, np.roll(zc2.sequence, shift))),
                np.sqrt(length),
            )
