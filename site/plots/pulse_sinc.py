import komm

from ._pulses import smooth


def plot():
    pulse = komm.SincPulse()
    return smooth(pulse, step=1)
