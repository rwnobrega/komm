import numpy as np
import pytest

import komm


def test_polar_code_invalid_construction():
    with pytest.raises(ValueError, match="'mu' must be at least 1"):
        komm.PolarCode(-1, [])
    with pytest.raises(ValueError, match="'mu' must be at least 1"):
        komm.PolarCode(0, [])
    with pytest.raises(TypeError, match="'mu' must be an integer"):
        komm.PolarCode(2.0, [0])  # type: ignore
    with pytest.raises(ValueError, match="elements of 'frozen' must be in"):
        komm.PolarCode(2, [3, 4])
    with pytest.raises(ValueError, match="elements of 'frozen' must be in"):
        komm.PolarCode(2, [-1, 0])
    with pytest.raises(TypeError, match="'frozen' must contain only integers"):
        komm.PolarCode(2, [0.5, 1.7])
    with pytest.raises(ValueError, match="elements of 'frozen' must be unique"):
        komm.PolarCode(2, [0, 0])


def test_polar_code_encode():
    # "LDPC and Polar Codes in 5G Standard"
    # Video "MATLAB Implementation for Encoding Polar Codes"
    # Indian Institute of Technology Madras (IITM), via NPTEL
    # Lecturer: Prof. Andrew Thangaraj
    # https://youtu.be/9b2z6bua0xY?t=862
    Q1 = [1, 2, 3, 5, 9, 4, 6, 10, 7, 11, 13, 8, 12, 14, 15, 16]
    frozen = np.array(Q1[:8]) - 1  # 0-indexed
    active = np.array(Q1[8:]) - 1  # 0-indexed
    np.testing.assert_equal(frozen, [0, 1, 2, 4, 8, 3, 5, 9])
    np.testing.assert_equal(active, [6, 10, 12, 7, 11, 13, 14, 15])
    code = komm.PolarCode(4, frozen)
    # https://youtu.be/9b2z6bua0xY?t=885 - msg = 1 1 0 1 1 0 0 1
    u = [1, 1, 1, 1, 0, 0, 0, 1]  # reordered according to 'active'
    # https://youtu.be/9b2z6bua0xY?t=1196 - video calls the codeword "u"
    v = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
    np.testing.assert_equal(code.encode(u), v)
