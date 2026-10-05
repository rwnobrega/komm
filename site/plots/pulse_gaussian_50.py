import komm

from ._pulses import smooth


def plot():
    pulse = komm.GaussianPulse(half_power_bandwidth=0.5)
    return smooth(pulse, step=0.5)
