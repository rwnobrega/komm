import numpy as np

import komm

from . import label, xticks, yticks
from ._constellations import angle, dots, plane


def plot():
    const = komm.ASKConstellation(
        order=4,
        base_amplitude=2 * np.sqrt(2),
        phase_offset=1 / 8,
    )
    points = const.matrix.ravel()
    fig, ax = plane((-4, 8), (-4, 8), 16)
    ticks = [-2, 2, 4, 6]
    xticks(ax, ticks)
    yticks(ax, ticks)
    angle(ax, points[-1], 2, r"$\tau/8$")
    dots(ax, points, ["above left"] * len(points))
    for v in ticks:
        label(ax, (v, 0), v, "below")
        label(ax, (0, v), v, "left")
    return fig
