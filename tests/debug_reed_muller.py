import numpy as np

import komm

code = komm.ReedMullerCode(2, 4)
awgn = komm.AWGNChannel(snr=4.0)
bpsk = komm.PAModulation(2)

print(code)
print((code.length, code.dimension, code.minimum_distance))
print(code._registered_encoders.keys())
print(code._registered_decoders.keys())
print(code._default_encoder)
print(code._default_decoder)
# print(code.generator_matrix)
# for partitions in code._reed_partitions:
#    print(partitions)
# print(code.parity_check_matrix)

n_words = 1_000
decoding_method = "weighted_reed"
input_type = code._registered_decoders[decoding_method]["input_type"]
message = np.random.randint(2, size=n_words * code.dimension)
codeword = code.encode(message)
sentword = bpsk.modulate(codeword)
recvword = awgn(sentword)
demodulated = bpsk.demodulate(recvword, input_type)
message_hat = code.decode(demodulated, method=decoding_method)

# print(message)
# print(codeword)
# print(recvword_hard)
# print((codeword != recvword_hard).astype(int))
print(f"{np.count_nonzero(message != message_hat)} / {len(message)}")
