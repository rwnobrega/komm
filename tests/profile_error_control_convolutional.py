import numpy as np
import komm

# ---- Parameters
feedforward_polynomials = [[0o117, 0o155]]
L = 1000
snr = 2.0
decoding_method = 'viterbi_soft'
# ----

code = komm.TerminatedConvolutionalCode(feedforward_polynomials, num_blocks=L)
awgn = komm.AWGNChannel(snr=snr)
bpsk = komm.PAModulation(2)
soft_or_hard = getattr(code, '_decode_' + decoding_method).input_type

message = np.random.randint(2, size=L)
codeword = code.encode(message)
sentword = bpsk.modulate(codeword)
recvword = awgn(sentword)
demodulated = bpsk.demodulate(recvword, decision_method=soft_or_hard)
message_hat = code.decode(demodulated, method=decoding_method)

print(f'{np.count_nonzero(message != message_hat)} / {len(message)}')
