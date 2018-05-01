import numpy as np
import komm

generator_matrix = [[0o7, 0o5]]
#generator_matrix = [[0o171, 0o133]]
#generator_matrix = [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]]

code = komm.ConvolutionalCode(generator_matrix)
awgn = komm.AWGNChannel(snr=2.0)
bpsk = komm.PAModulation(2)

k, n, nu = code.num_input_bits, code.num_output_bits, code.overall_constraint_length
print(code)
print((n, k, nu))

L = 10000
decoding_method = 'viterbi_hard'
input_type = code._registered_decoders[decoding_method]['input_type']

message = np.random.randint(2, size=k*L)
message_tail = np.concatenate((message, np.zeros(k*nu, dtype=np.int)))
codeword = code.encode(message_tail)
sentword = bpsk.modulate(codeword)
recvword = awgn(sentword)

demodulated = bpsk.demodulate(recvword, input_type)
message_tail_hat = code.decode(demodulated, method=decoding_method)
message_hat = message_tail_hat[:-k*nu]

#print(f'message = {message} {len(message)}')
#print(f'message_tail = {message_tail} {len(message_tail)}')
#print(f'codeword = {codeword} {len(codeword)}')
#print(f'recvword = {recvword} {len(recvword)}')
#print(f'message_tail_hat = {message_tail_hat} {len(message_tail_hat)}')
#print(f'message_hat = {message_hat} {len(message_hat)}')
print(f'{np.count_nonzero(message != message_hat)} / {len(message)}')
