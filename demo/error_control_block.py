import numpy as np
import matplotlib.pylab as plt
import multiprocessing
import itertools

import komm


#code = komm.ReedMullerCode(2, 6)
code = komm.HammingCode(3)
#code = komm.GolayCode()
#code = komm.CyclicCode(length=7, generator_polynomial=0o13)  # Hamming(3)
#code = komm.CyclicCode(length=23, generator_polynomial=0o5343)  # Golay()
#code = komm.BCHCode(5, 5)
awgn = komm.AWGNChannel()
bpsk = komm.PAModulation(2)

k, n, d = code.dimension, code.length, code.minimum_distance
code_rate = k / n

print(code)
print((n, k, d))

methods = [
#    'berlekamp',
#    'meggitt',
    'syndrome_table',
    'exhaustive_search_soft',
#    'reed',
#    'weighted_reed'
]

ebno_dB_list = np.arange(-1.0, 9.0, step=1.0)
ebno_list = 10**(ebno_dB_list/10)

L = 1000
message = np.random.randint(2, size=k*L)
codeword = code.encode(message)
sentword = bpsk.modulate(codeword)

max_bit_errors = 1_000
max_bits = 1_000_000

def simulate(ebno_dB, method):
    ebno = 10**(ebno_dB/10)
    awgn.snr = 2 * ebno * code_rate
    num_bit_errors = 0
    num_word_errors = 0
    num_bits = 0
    while num_bit_errors < max_bit_errors and num_bits < max_bits:
        recvword = awgn(sentword)
        soft_or_hard = getattr(code, '_decode_' + method).input_type
        demodulated = bpsk.demodulate(recvword, decision_method=soft_or_hard)
        message_hat = code.decode(demodulated, method=method)
        num_bits += k*L
        num_bit_errors += np.count_nonzero(message_hat != message)
        num_word_errors += np.sum(np.any(np.reshape(message_hat, newshape=(-1, k)) != np.reshape(message, newshape=(-1, k)), axis=1))
    print(f'{ebno_dB} dB - {method} - {num_bit_errors}/{max_bit_errors} - {num_bits}/{max_bits}')
    return num_bit_errors / num_bits, num_word_errors / (num_bits/k)

params = list(itertools.product(ebno_dB_list, methods))
with multiprocessing.Pool(8) as pool:
    results = pool.starmap(simulate, params)

ber = {method: [] for method in methods}
wer = {method: [] for method in methods}
for (_, method), (bber, wwer) in zip(params, results):
    ber[method].append(bber)
    wer[method].append(wwer)

plt.close('all')
plt.figure(1)
for method in methods:
    plt.semilogy(ebno_dB_list, ber[method])
plt.semilogy(ebno_dB_list, komm.util.qfunc(np.sqrt(2*ebno_list)), 'k--')
plt.legend(methods)
plt.title(f'Bit error rate for {code}')
plt.xlabel('Eb / No [dB]')
plt.ylabel('BER')
plt.grid(True)
plt.figure(2)
for method in methods:
    plt.semilogy(ebno_dB_list, wer[method])
plt.legend(methods)
plt.title(f'Word error rate for {code}')
plt.xlabel('Eb / No [dB]')
plt.ylabel('WER')
plt.grid(True)
plt.show()
