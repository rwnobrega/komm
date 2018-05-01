import datetime
import numpy as np
import matplotlib.pylab as plt
import multiprocessing
import itertools

import komm


code = komm.ConvolutionalCode.optimum_convolutional_code(2, 1, 2)
awgn = komm.AWGNChannel()
bpsk = komm.PAModulation(order=2)

L = 1000
n, k, nu = code.num_output_bits, code.num_input_bits, code.overall_constraint_length
code_rate = (L / (L + nu)) * (k / n)

print(code)
print((n, k, nu))

methods = ['viterbi_hard', 'viterbi_soft']

ebno_dB_list = np.arange(-1.0, 7.0, step=1.0)
ebno_list = 10**(ebno_dB_list/10)

message = np.random.randint(2, size=L*k)
message_tail = np.concatenate((message, np.zeros(k*nu, dtype=np.int)))
codeword = code.encode(message_tail)
sentword = bpsk.modulate(codeword)

max_bit_errors = 1_000
max_bits = 1_000_000

def simulate(ebno_dB, method):
    ebno = 10**(ebno_dB/10)
    awgn.snr = 2 * ebno * code_rate
    num_bit_errors = 0
    num_bits = 0
    while num_bit_errors < max_bit_errors and num_bits < max_bits:
        recvword = awgn(sentword)
        soft_or_hard = getattr(code, '_decode_' + method).input_type
        demodulated = bpsk.demodulate(recvword, decision_method=soft_or_hard)
        message_tail_hat = code.decode(demodulated, method=method)
        message_hat = message_tail_hat[:-k*nu]
        num_bits += k*L
        num_bit_errors += np.count_nonzero(message_hat != message)
    print(f'{ebno_dB} dB - {method} - {num_bit_errors}/{max_bit_errors} - {num_bits}/{max_bits}')
    return num_bit_errors / num_bits

params = list(itertools.product(ebno_dB_list, methods))
with multiprocessing.Pool(8) as pool:
    results = pool.starmap(simulate, params)

ber = {method: [] for method in methods}
for (_, method), bber in zip(params, results):
    ber[method].append(bber)

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
# ~plt.savefig(str(datetime.datetime.now()).replace(':', '_')[:-7] + '.png')
plt.show()
