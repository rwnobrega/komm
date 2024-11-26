import itertools as it

import numpy as np
import pytest

import komm


@pytest.mark.parametrize("length", (2, 4, 8, 16, 32, 64))
def test_walsh_hadamard_index_0(length):
    walsh_hadamard = komm.WalshHadamardSequence(length, index=0)
    np.testing.assert_array_equal(
        walsh_hadamard.polar_sequence,
        [1] * length,
    )


@pytest.mark.parametrize("length", (2, 4, 8, 16, 32, 64))
def test_walsh_hadamard_index_1(length):
    walsh_hadamard = komm.WalshHadamardSequence(length, index=1)
    np.testing.assert_array_equal(
        walsh_hadamard.polar_sequence,
        [1, -1] * (length // 2),
    )


@pytest.mark.parametrize("length", (4, 8, 16, 32, 64))
def test_walsh_hadamard_index_2(length):
    walsh_hadamard = komm.WalshHadamardSequence(length, index=2)
    np.testing.assert_array_equal(
        walsh_hadamard.polar_sequence,
        [1, 1, -1, -1] * (length // 4),
    )


@pytest.mark.parametrize("length", (2, 4, 8, 16, 32, 64))
@pytest.mark.parametrize("ordering", ("natural", "sequency"))
def test_walsh_hadamard_orthogonality(length, ordering):
    walsh_hadamard = []
    for i in range(length):
        walsh_hadamard.append(komm.WalshHadamardSequence(length, ordering, index=i))
    for i1, i2 in it.combinations(range(length), 2):
        seq1 = walsh_hadamard[i1].polar_sequence
        seq2 = walsh_hadamard[i2].polar_sequence
        assert np.correlate(seq1, seq2) == (length if i1 == i2 else 0)


@pytest.mark.parametrize("length", (2, 4, 8, 16, 32, 64))
def test_walsh_hadamard_sequency_sign_changes(length):
    # Row $i$ must have exactly $i$ sign changes
    for index in range(length):
        walsh_hadamard = komm.WalshHadamardSequence(length, "sequency", index=index)
        assert np.sum(np.abs(np.diff(walsh_hadamard.bit_sequence))) == index


def test_walsh_hadamard_invalid():
    with pytest.raises(ValueError):
        komm.WalshHadamardSequence(3)
    with pytest.raises(ValueError):
        komm.WalshHadamardSequence(4, index=4)
    with pytest.raises(ValueError):
        komm.WalshHadamardSequence(4, ordering="invalid")  # type: ignore
    with pytest.raises(NotImplementedError):
        komm.WalshHadamardSequence(4, ordering="dyadic")
