import numpy as np
import pytest

import komm
from komm._modulation.labelings import cartesian_product


def test_qam_constellations():
    np.testing.assert_allclose(
        komm.QAModulation(4).constellation,
        [
            -1.0 - 1.0j,
            1.0 - 1.0j,
            -1.0 + 1.0j,
            1.0 + 1.0j,
        ],
    )
    np.testing.assert_allclose(
        komm.QAModulation((4, 2)).constellation,
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
    np.testing.assert_allclose(
        komm.QAModulation(16).constellation,
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
def test_qam_vs_pam(orders, base_amplitudes):
    (M_I, M_Q) = orders
    (A_I, A_Q) = base_amplitudes
    qam = komm.QAModulation(orders, base_amplitudes=base_amplitudes)

    pam_I = komm.PAModulation(M_I, base_amplitude=A_I)
    pam_Q = komm.PAModulation(M_Q, base_amplitude=A_Q)
    qam_real_2d = cartesian_product(
        pam_I.constellation.reshape(-1, 1), pam_Q.constellation.reshape(-1, 1)
    )
    qam_complex_1d = qam_real_2d[:, 0] + 1j * qam_real_2d[:, 1]
    np.testing.assert_allclose(qam_complex_1d, qam.constellation)

    qam_doc_formula = []
    for i in range(M_I * M_Q):
        i_I, i_Q = i % M_I, i // M_I
        qam_doc_formula.append(
            A_I * (2 * i_I - M_I + 1) + 1j * A_Q * (2 * i_Q - M_Q + 1)
        )
    np.testing.assert_allclose(qam_doc_formula, qam.constellation)


def test_qam_invalid():
    with pytest.raises(ValueError, match="must be a square power of two"):
        komm.QAModulation(8)
    with pytest.raises(ValueError, match="must be a power of two"):
        komm.QAModulation((6, 2))
    with pytest.raises(ValueError, match="must be a power of two"):
        komm.QAModulation((3, 3))
    with pytest.raises(ValueError, match="if string, 'labeling' must be in"):
        komm.QAModulation(4, labeling="invalid")
    with pytest.raises(ValueError, match="shape of 'labeling' must be"):
        komm.QAModulation(4, labeling=[[0, 0], [1, 0], [1, 1]])
