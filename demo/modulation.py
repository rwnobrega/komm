import itertools
import multiprocessing

import numpy as np
import matplotlib.pylab as plt

import komm


awgn = komm.AWGNChannel()

num_symbols = 100_000
ebno_dB_list = np.arange(-1.0, 9.0, step=1.0)
ebno_list = 10**(ebno_dB_list/10)

mod_dict = {
    '2-pam': komm.PAModulation(2),
    'bpsk': komm.PSKModulation(2),
    'qpsk': komm.PSKModulation(4),
#    'qpsk-nat': komm.PSKModulation(4, labeling='natural'),
#    '8-psk': komm.PSKModulation(8),
#    '8-psk-nat': komm.PSKModulation(8, labeling='natural'),
#    '16-qam-refl-2d': komm.QAModulation(16, labeling='reflected_2d'),
#    '16-qam-nat': komm.QAModulation(16, labeling='natural'),
#    '16-qam-refl': komm.QAModulation(16, labeling='reflected'),
}


def simulate(ebno_dB, mod_key):
    mod = mod_dict[mod_key]
    if isinstance(mod, komm.RealModulation):
        mod_dimension = 1
    elif isinstance(mod, komm.ComplexModulation):
        mod_dimension = 2
    ebno = 10**(ebno_dB/10)
    awgn.snr = 2 * ebno * mod.bits_per_symbol / mod_dimension
    awgn.signal_power = mod.energy_per_symbol
    num_bits = mod.bits_per_symbol * num_symbols
    bits = np.random.randint(2, size=num_bits)
    sentword = mod.modulate(bits)
    recvword = awgn(sentword)
    bits_hat = mod.demodulate(recvword)
    print(f'{ebno_dB} - {mod}')
    return np.count_nonzero(bits_hat != bits) / num_bits

params = list(itertools.product(ebno_dB_list, mod_dict.keys()))
with multiprocessing.Pool(8) as pool:
    results = pool.starmap(simulate, params)

ber = {mod_key: [] for mod_key in mod_dict.keys()}
for (_, mod_key), bber in zip(params, results):
    ber[mod_key].append(bber)


#plt.close('all')
#plt.scatter(recvword.real, recvword.imag, c='r')
#plt.scatter(mod.constellation.real, mod.constellation.imag, c='b')
#plt.title(repr(mod))
#plt.xlabel('Re')
#plt.ylabel('Im')
#plt.axis('equal')
#plt.grid('on')
#plt.show()

plt.close('all')
plt.figure(1)
for mod_key in mod_dict.keys():
    plt.semilogy(ebno_dB_list, ber[mod_key])
plt.semilogy(ebno_dB_list, komm.util.qfunc(np.sqrt(2*ebno_list)), 'k--')
plt.legend(mod_dict.keys())
plt.title(f'Bit error rate')
plt.xlabel('Eb / No [dB]')
plt.ylabel('BER')
plt.grid(True)
plt.show()
