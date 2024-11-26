import numpy as np
import pytest

import komm


@pytest.mark.parametrize("length", [2, 3, 4, 5, 7, 11, 13])
def test_barker(length):
    barker = komm.BarkerSequence(length)
    assert barker.length == length
    acorr = barker.autocorrelation()
    assert acorr[0] == length
    assert np.all(np.abs(acorr[1:]) <= 1)


@pytest.mark.parametrize("length", [-10, 0, 1, 6, 8, 9, 10, 12, 14])
def test_barker_invalid_length(length):
    with pytest.raises(ValueError):
        komm.BarkerSequence(length)
