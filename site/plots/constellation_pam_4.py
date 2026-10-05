import komm

from ._constellations import pam


def plot():
    const = komm.PAMConstellation(4)
    return pam(const)
