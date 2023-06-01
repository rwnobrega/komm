import numpy as np

import komm

# code = komm.CyclicCode(length=7, generator_polynomial=0o13)  # Hamming(3)
code = komm.CyclicCode(length=23, generator_polynomial=0o5343)  # Golay()
# code = komm.CyclicCode(length=127, generator_polynomial=0o3447023271); code._minimum_distance = 9
awgn = komm.AWGNChannel(snr=4.0)
bpsk = komm.PAModulation(2)

print(code)
print((code.length, code.dimension, code.minimum_distance))
print(code._available_decoding_methods())
print(code._default_encoder())
print(code._default_decoder(float))
print(code.generator_polynomial)
# print(code.generator_matrix)
print(code.parity_polynomial)
# print(code.parity_check_matrix)

n_words = 10_000

message = np.random.randint(2, size=n_words * code.dimension)
codeword = code.encode(message)

sentword = bpsk.modulate(codeword)
recvword = awgn(sentword)
demodulated_hard = bpsk.demodulate(recvword)
message_hat = code.decode(demodulated_hard, method="meggitt")

# print(message)
# print(codeword)
# print(recvword_hard)
# print((codeword != recvword_hard).astype(int))
print(f"{np.count_nonzero(message != message_hat)} / {len(message)}")
