import numpy as np
import matplotlib.pylab as plt
import multiprocessing
import itertools

import komm


code = komm.HammingCode(3)
awgn = komm.AWGNChannel()

k, n, d = code.dimension, code.length, code.minimum_distance
code_rate = k / n

print(code)
print((n, k, d))

methods = ['syndrome_table', 'exhaustive_search_soft']
modulations = {
    'BPSK': komm.PSKModulation(2),
    'QPSK': komm.PSKModulation(4),
    '8-PSK': komm.PSKModulation(8),
}

ebno_dB_list = np.arange(0.0, 7.0, step=1.0)
ebno_list = 10.0**(ebno_dB_list / 10.0)

L = 3000
message = np.random.randint(2, size=k*L)
codeword = code.encode(message)

max_bit_errors = 10_000
max_bits = 10_000_000

def simulate(ebno_dB, method, mod_key):
    ebno = 10.0**(ebno_dB / 10.0)
    mod = modulations[mod_key]
    awgn.snr = ebno * code_rate * mod.bits_per_symbol
    awgn.signal_power = mod.energy_per_symbol
    mod.channel_snr = awgn.snr
    num_bit_errors = 0
    num_bits = 0
    sentword = mod.modulate(codeword)
    recvword = awgn(sentword)
    while num_bit_errors < max_bit_errors and num_bits < max_bits:
        soft_or_hard = getattr(code, '_decode_' + method).input_type
        demodulated = mod.demodulate(recvword, decision_method=soft_or_hard)
        message_hat = code.decode(demodulated, method=method)
        num_bits += k*L
        num_bit_errors += np.count_nonzero(message_hat != message)
    print(f'{ebno_dB} dB - {method} - {mod_key} - {num_bit_errors}/{max_bit_errors} - {num_bits}/{max_bits}')
    return num_bit_errors / num_bits

params = list(itertools.product(ebno_dB_list, methods, modulations.keys()))
with multiprocessing.Pool(8) as pool:
    results = pool.starmap(simulate, params)

ber = {(method, modulation): [] for method in methods for modulation in modulations}
for (_, method, modulation), bber in zip(params, results):
    ber[method, modulation].append(bber)

plt.close('all')

for i, modulation in enumerate(modulations):
    plt.figure(i + 1)
    for method in methods:
        plt.semilogy(ebno_dB_list, ber[method, modulation])
    plt.semilogy(ebno_dB_list, komm.util.qfunc(np.sqrt(2*ebno_list)), 'k--')
    plt.legend(methods)
    plt.title(f'BER for {code}, {modulation}')
    plt.xlabel('Eb / No [dB]')
    plt.ylabel('BER')
    plt.grid(True)

plt.show()
