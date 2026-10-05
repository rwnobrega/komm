import komm

from . import label, xticks, yticks
from ._constellations import dots, plane


def plot():
    const = komm.ASKConstellation(4)
    points = const.matrix.ravel()
    fig, ax = plane((-1.5, 4), (-1.5, 1.5), 48)
    xticks(ax, [-1])
    yticks(ax, [-1, 1])
    dots(ax, points, ["above right"] + ["above"] * 3)
    for x in points.real:
        label(ax, (x, 0), x, "below right" if x == 0 else "below")
    return fig
