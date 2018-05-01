import numpy as np
import komm

dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2], [0.5, 0.05, 0.45]])
print(dmc)
print(dmc.capacity(max_iters=1, error_tolerance=1e-1))
input_sequence = np.random.choice(dmc.input_cardinality, size=10)
output_sequence = dmc(input_sequence)
print(input_sequence, output_sequence)

bsc1 = komm.DiscreteMemorylessChannel([[0.75, 0.25], [0.25, 0.75]])
bsc2 = komm.BinarySymmetricChannel(0.25)
print(bsc1)
print(bsc2)
print(bsc1.capacity(), bsc2.capacity())
input_sequence = np.random.randint(2, size=10)
output_sequence = bsc2(input_sequence)
print(input_sequence, output_sequence)

bec = komm.BinaryErasureChannel(0.25)
print(bec)
print(bec.capacity())
input_sequence = np.random.randint(2, size=30)
output_sequence = bec(input_sequence)
print(input_sequence)
print(output_sequence)
