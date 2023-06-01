import numpy as np

import komm

code = komm.BCHCode(4, 2)
bpsk = komm.PAModulation(2)
awgn = komm.AWGNChannel(2.0)

print(code)
print((code.length, code.dimension, code.minimum_distance))
print(code._available_decoding_methods())
print(code._default_encoder())
print(code._default_decoder(float))
print(code.generator_polynomial)
# print(code.generator_matrix)
print(code.parity_polynomial)
# print(code.parity_check_matrix)

n_words = 1_000

message = np.random.randint(2, size=n_words * code.dimension)
codeword = code.encode(message)

sentword = bpsk.modulate(codeword)
recvword = awgn(sentword)
demodulated_hard = bpsk.demodulate(recvword)
message_hat = code.decode(demodulated_hard)

# print(message)
# print(codeword)
# print(demodulated_hard)
# print((codeword != recvword_hard).astype(int))
print(f"{np.count_nonzero(message != message_hat)} / {len(message)}")
