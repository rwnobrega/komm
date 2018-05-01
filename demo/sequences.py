import numpy as np
import matplotlib.pylab as plt

import komm


def barker_demo():
    L = 13
    bark = komm.BarkerSequence(length=L)
    seq = (-1)**bark.sequence
    plt.close('all')
    plt.figure()
    plt.stem(np.arange(L), seq)
    plt.title(repr(bark) + ' - ' + str(bark.sequence.tolist()))
    plt.show()


def walsh_hadamard_demo():
    had = komm.WalshHadamardSequence(length=32, index=29)
    print(had)
    print(had.sequence)


def lfsr_demo():
    m = 5
    L = 2**m - 1
    lfsr = komm.LFSRSequence(0b100101)
    seq = (-1)**lfsr.sequence

    shifts = np.arange(-2*L + 1, 2*L)
    cyclic_acorr = np.empty_like(shifts, dtype=np.float)
    for (i, ell) in enumerate(shifts):
        cyclic_acorr[i] = np.dot(seq, np.roll(seq, ell)) / L

    plt.close('all')
    plt.figure()
    plt.stem(np.arange(L), seq)
    plt.title(repr(lfsr))
    plt.figure()
    plt.plot(shifts, cyclic_acorr)
    plt.title('Cyclic autocorrelation')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
#    barker_demo()
#    walsh_hadamard_demo()
    lfsr_demo()
