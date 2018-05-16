import numpy as np
import matplotlib.pylab as plt

import komm


def barker_demo():
    bark = komm.BarkerSequence(length=13)
    plt.close('all')
    plt.figure()
    plt.stem(np.arange(bark.length), bark.polar_sequence)
    plt.title(repr(bark))
    plt.show()


def walsh_hadamard_demo():
    had = komm.WalshHadamardSequence(length=32, index=29)
    plt.close('all')
    plt.figure()
    plt.stem(np.arange(had.length), had.polar_sequence)
    plt.title(repr(had))
    plt.show()


def lfsr_demo():
    lfsr = komm.LFSRSequence(0b100101)
    shifts = np.arange(-2*lfsr.length + 1, 2*lfsr.length)
    cyclic_acorr = lfsr.cyclic_autocorrelation(shifts)

    plt.close('all')
    plt.figure()
    plt.stem(range(lfsr.length), lfsr.polar_sequence)
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
