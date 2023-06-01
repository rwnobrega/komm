import numpy as np

import komm

awgn = komm.AWGNChannel(snr=12.0, signal_power="measured")

power = lambda v: np.real(np.vdot(v, v)) / len(v)

# s = 2.0 * (-1.0)**np.random.randint(2, size=100000)
s = 2.0 * (-1.0) ** np.random.randint(2, size=100000) + 2.0j * (-1.0) ** np.random.randint(2, size=100000)
r = awgn(s)
print(s)
print(r)
print(power(s))
print(power(r))
print(power(r - s))
print(power(s) / power(r - s))
