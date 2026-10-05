import komm

from ._constellations import angle, rings


def plot():
    const = komm.PSKConstellation(8, amplitude=0.5, phase_offset=1 / 16)
    fig, ax = rings(const, ticks=[0.5])
    angle(ax, const.matrix[0, 0], 1 / 6, r"$\tau/16$")
    return fig
