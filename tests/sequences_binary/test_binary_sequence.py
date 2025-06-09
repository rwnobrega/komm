import numpy as np

import komm


def test_binary_sequence_construction():
    sequence = komm.BinarySequence(bit_sequence=[0, 1, 1, 0])
    np.testing.assert_equal(sequence.bit_sequence, [0, 1, 1, 0])
    np.testing.assert_equal(sequence.polar_sequence, [1, -1, -1, 1])
    assert sequence.length == 4


def test_binary_sequence_autocorrelation():
    sequence = komm.BinarySequence(bit_sequence=[0, 1, 1, 0])
    np.testing.assert_equal(
        sequence.autocorrelation(),
        [4, -1, -2, 1],
    )
    np.testing.assert_equal(
        sequence.autocorrelation(normalized=True),
        [1, -0.25, -0.5, 0.25],
    )
    np.testing.assert_equal(
        sequence.autocorrelation(shifts=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        [0, 0, 1, -2, -1, 4, -1, -2, 1, 0, 0],
    )
    np.testing.assert_equal(
        sequence.autocorrelation(shifts=np.arange(-5, 6)),
        [0, 0, 1, -2, -1, 4, -1, -2, 1, 0, 0],
    )
    np.testing.assert_equal(
        sequence.autocorrelation(shifts=np.arange(-5, 6), normalized=True),
        [0, 0, 0.25, -0.5, -0.25, 1, -0.25, -0.5, 0.25, 0, 0],
    )


def test_binary_sequence_cyclic_autocorrelation():
    sequence = komm.BinarySequence(bit_sequence=[0, 1, 1, 0])
    np.testing.assert_equal(
        sequence.cyclic_autocorrelation(),
        [4, 0, -4, 0],
    )
    np.testing.assert_equal(
        sequence.cyclic_autocorrelation(normalized=True),
        [1, 0, -1, 0],
    )
    np.testing.assert_equal(
        sequence.cyclic_autocorrelation(shifts=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        [0, 4, 0, -4, 0, 4, 0, -4, 0, 4, 0],
    )
    np.testing.assert_equal(
        sequence.cyclic_autocorrelation(shifts=np.arange(-5, 6)),
        [0, 4, 0, -4, 0, 4, 0, -4, 0, 4, 0],
    )
    np.testing.assert_equal(
        sequence.cyclic_autocorrelation(shifts=np.arange(-5, 6), normalized=True),
        [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0],
    )
