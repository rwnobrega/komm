import numpy as np
import komm

# ---- Parameters
feedforward_polynomials = [[0o117, 0o155]]
L = 1000
snr = 2.0
decoding_method = 'viterbi_soft'
# ----

code = komm.ConvolutionalCode(feedforward_polynomials)
awgn = komm.AWGNChannel(snr=snr)
bpsk = komm.PAModulation(2)
k, n, nu = code.num_input_bits, code.num_output_bits, code.overall_constraint_length
soft_or_hard = getattr(code, '_decode_' + decoding_method).input_type

message = np.random.randint(2, size=k*L)
message_tail = np.concatenate((message, np.zeros(k*nu, dtype=np.int)))
codeword = code.encode(message_tail)
sentword = bpsk.modulate(codeword)
recvword = awgn(sentword)
demodulated = bpsk.demodulate(recvword, decision_method=soft_or_hard)
message_tail_hat = code.decode(demodulated, method=decoding_method)
message_hat = message_tail_hat[:-k*nu]

print(f'{np.count_nonzero(message != message_hat)} / {len(message)}')
