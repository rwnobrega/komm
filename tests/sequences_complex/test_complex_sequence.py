import pytest

import komm


def test_complex_sequence_invalid_construction():
    with pytest.raises(ValueError, match="'sequence' must be a 1D-array"):
        komm.ComplexSequence([[1, 1j], [-1, -1j]])
