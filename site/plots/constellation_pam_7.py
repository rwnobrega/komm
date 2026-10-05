import komm

from ._constellations import pam


def plot():
    const = komm.PAMConstellation(7, delta=5)
    return pam(const)
