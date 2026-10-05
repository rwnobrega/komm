import komm

from ._pulses import smooth


def plot():
    pulse = komm.RaisedCosinePulse(rolloff=0.25)
    return smooth(pulse, step=1)
