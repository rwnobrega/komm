import itertools as it

import numpy as np
import pytest

import komm


@pytest.mark.parametrize("length", (2, 4, 8, 16, 32, 64))
def test_walsh_hadamard_1(length):
    walsh_hadamard = komm.WalshHadamardSequence(length, index=0)
    assert np.array_equal(walsh_hadamard.polar_sequence, [1] * length)
    walsh_hadamard = komm.WalshHadamardSequence(length, index=1)
    assert np.array_equal(walsh_hadamard.polar_sequence, [1, -1] * (length // 2))
    if length > 2:
        walsh_hadamard = komm.WalshHadamardSequence(length, index=2)
        assert np.array_equal(
            walsh_hadamard.polar_sequence, [1, 1, -1, -1] * (length // 4)
        )


@pytest.mark.parametrize("length", (2, 4, 8, 16, 32, 64))
@pytest.mark.parametrize("ordering", ("natural", "sequency"))
def test_walsh_hadamard_2(length, ordering):
    # Test that the sequences are orthogonal
    walsh_hadamard = []
    for i in range(length):
        walsh_hadamard.append(komm.WalshHadamardSequence(length, ordering, index=i))
    for i1, i2 in it.combinations(range(length), 2):
        seq1 = walsh_hadamard[i1].polar_sequence
        seq2 = walsh_hadamard[i2].polar_sequence
        assert np.correlate(seq1, seq2) == (length if i1 == i2 else 0)


@pytest.mark.parametrize("length", (2, 4, 8, 16, 32, 64))
def test_walsh_hadamard_3(length):
    # Test that the 'sequency' ordering has the property that row $i$ has exactly $i$ sign changes
    for index in range(length):
        walsh_hadamard = komm.WalshHadamardSequence(length, "sequency", index=index)
        assert np.sum(np.abs(np.diff(walsh_hadamard.bit_sequence))) == index
