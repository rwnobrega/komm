import komm

from ._constellations import rings


def plot():
    const = komm.APSKConstellation(orders=(4, 4), amplitudes=(1.0, 2.0))
    fig, _ = rings(const, ticks=[1, 2])
    return fig
