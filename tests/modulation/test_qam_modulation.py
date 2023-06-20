import numpy as np

import komm


def test_qam_modulation():
    qam4 = komm.QAModulation(4)
    qam8 = komm.QAModulation((4, 2))
    qam16 = komm.QAModulation(16)
    assert np.allclose(
        qam4.constellation,
        [
            -1.0 - 1.0j,
            1.0 - 1.0j,
            -1.0 + 1.0j,
            1.0 + 1.0j,
        ],
    )
    assert np.allclose(
        qam8.constellation,
        [
            -3.0 - 1.0j,
            -1.0 - 1.0j,
            1.0 - 1.0j,
            3.0 - 1.0j,
            -3.0 + 1.0j,
            -1.0 + 1.0j,
            1.0 + 1.0j,
            3.0 + 1.0j,
        ],
    )
    assert np.allclose(
        qam16.constellation,
        [
            -3.0 - 3.0j,
            -1.0 - 3.0j,
            1.0 - 3.0j,
            3.0 - 3.0j,
            -3.0 - 1.0j,
            -1.0 - 1.0j,
            1.0 - 1.0j,
            3.0 - 1.0j,
            -3.0 + 1.0j,
            -1.0 + 1.0j,
            1.0 + 1.0j,
            3.0 + 1.0j,
            -3.0 + 3.0j,
            -1.0 + 3.0j,
            1.0 + 3.0j,
            3.0 + 3.0j,
        ],
    )
