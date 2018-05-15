import numpy as np

import komm


def test_ask_modulation():
    ask2 = komm.ASKModulation(2)
    ask4 = komm.ASKModulation(4)
    ask8 = komm.ASKModulation(8)

    assert np.allclose(ask2.constellation, [0.0, 1.0])
    assert np.allclose(ask4.constellation, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(ask8.constellation, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    x = [-0.5, 0.25, 0.4, 0.65, 2.1, 10.0]
    assert np.allclose(ask2.demodulate(x), ask2.symbols_to_bits([0, 0, 0, 1, 1, 1]))
    assert np.allclose(ask4.demodulate(x), ask4.symbols_to_bits([0, 0, 0, 1, 2, 3]))
    assert np.allclose(ask8.demodulate(x), ask8.symbols_to_bits([0, 0, 0, 1, 2, 7]))


def test_psk_modulation():
    bpsk = komm.PSKModulation(2)
    qpsk = komm.PSKModulation(4)
    psk8 = komm.PSKModulation(8)
    r4 = (1.0 + 1.0j) / np.sqrt(2)
    assert np.allclose(bpsk.constellation, [1.0, -1.0])
    assert np.allclose(qpsk.constellation, [1.0, 1.0j, -1.0, -1.0j])
    assert np.allclose(psk8.constellation, [1.0, r4, 1.0j, -r4.conjugate(), -1.0, -r4, -1.0j, r4.conjugate()])


def test_qam_modulation():
    qam4 = komm.QAModulation(4)
    qam8 = komm.QAModulation((4,2))
    qam16 = komm.QAModulation(16)
    assert np.allclose(qam4.constellation, [-1.0 - 1.0j, 1.0 - 1.0j, -1.0 + 1.0j, 1.0 + 1.0j])
    assert np.allclose(qam8.constellation, [-3.0 - 1.0j, -1.0 - 1.0j, 1.0 - 1.0j, 3.0 - 1.0j,
                                            -3.0 + 1.0j, -1.0 + 1.0j, 1.0 + 1.0j, 3.0 + 1.0j])
    assert np.allclose(qam16.constellation, [-3.0 - 3.0j, -1.0 - 3.0j, 1.0 - 3.0j, 3.0 - 3.0j,
                                             -3.0 - 1.0j, -1.0 - 1.0j, 1.0 - 1.0j, 3.0 - 1.0j,
                                             -3.0 + 1.0j, -1.0 + 1.0j, 1.0 + 1.0j, 3.0 + 1.0j,
                                             -3.0 + 3.0j, -1.0 + 3.0j, 1.0 + 3.0j, 3.0 + 3.0j])
