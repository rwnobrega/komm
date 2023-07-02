import numpy as np
import pytest

import komm
from komm._util.matrices import cartesian_product


def test_qam_modulation_1():
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


@pytest.mark.parametrize(
    "orders, base_amplitudes",
    [
        ((4, 2), (1, 2)),
        ((4, 2), (3, 1)),
        ((2, 4), (10, 20)),
        ((2, 4), (15, 12)),
        ((4, 4), (3, 7)),
    ],
)
def test_qam_modulation_2(orders, base_amplitudes):
    (M_I, M_Q) = orders
    (A_I, A_Q) = base_amplitudes
    qam = komm.QAModulation(orders, base_amplitudes=base_amplitudes)

    pam_I = komm.PAModulation(M_I, base_amplitude=A_I)
    pam_Q = komm.PAModulation(M_Q, base_amplitude=A_Q)
    qam_real_2d = cartesian_product(pam_I.constellation, pam_Q.constellation)
    qam_complex_1d = qam_real_2d[:, 0] + 1j * qam_real_2d[:, 1]
    assert np.allclose(qam_complex_1d, qam.constellation)

    qam_doc_formula = []
    for i in range(M_I * M_Q):
        i_I, i_Q = i % M_I, i // M_I
        qam_doc_formula.append(A_I * (2 * i_I - M_I + 1) + 1j * A_Q * (2 * i_Q - M_Q + 1))
    assert np.allclose(qam_doc_formula, qam.constellation)
