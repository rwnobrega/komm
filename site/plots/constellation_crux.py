import komm

from . import canvas, label, xticks, yticks
from ._constellations import dots


def plot():
    const = komm.Constellation([[0, 4], [-2, 2], [2, 2], [1, 1], [0, -2]])
    points = const.matrix @ [1, 1j]
    fig, ax = canvas((-4, 5), (-4, 6), scale=(24, 24))
    xticks(ax, [-2, 2])
    yticks(ax, [-2, 2, 4])
    sides = ["above right", "above", "above", "above", "above right"]
    dots(ax, points, sides, symbol=r"\mathbf{x}")
    for v in [-2, 2]:
        label(ax, (v, 0), v, "below")
    for v in [-2, 2, 4]:
        label(ax, (0, v), v, "left")
    return fig
