import numpy as np

import komm


def test_psk_modulation():
    bpsk = komm.PSKModulation(2)
    qpsk = komm.PSKModulation(4)
    psk8 = komm.PSKModulation(8)
    r4 = (1.0 + 1.0j) / np.sqrt(2)
    assert np.allclose(bpsk.constellation, [1.0, -1.0])
    assert np.allclose(qpsk.constellation, [1.0, 1.0j, -1.0, -1.0j])
    assert np.allclose(psk8.constellation, [1.0, r4, 1.0j, -r4.conjugate(), -1.0, -r4, -1.0j, r4.conjugate()])
