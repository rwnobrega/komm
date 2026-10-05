import komm

from ._pulses import smooth


def plot():
    pulse = komm.BeaulieuPulse(rolloff=0.75)
    return smooth(pulse, step=1)
