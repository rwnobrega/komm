import numpy as np

from . import DASHES, canvas, label, line, text, xticks, yticks


def plane(xlim, ylim, scale):
    fig, ax = canvas(xlim, ylim, scale=(scale, scale))
    text(ax, (xlim[1], 0), "Re", "right")
    text(ax, (0, ylim[1]), "Im", "above")
    return fig, ax


def dots(ax, points, sides):
    ax.plot(points.real, points.imag, "o", color="black", ms=6, mew=0)
    for i, (z, side) in enumerate(zip(points, sides)):
        text(ax, (z.real, z.imag), f"$x_{{{i}}}$", side)


def pam(const):
    points = const.matrix.ravel()
    step = const.minimum_distance()
    edge = points.max() + step
    scale = (64 / step, 64 / step)
    fig, ax = canvas((-edge, edge), (-step / 8, step / 8), scale, yaxis=False)
    xticks(ax, np.append(points - step / 2, points[-1] + step / 2))
    dots(ax, points, ["above"] * len(points))
    for x in points:
        label(ax, (x, 0), x, "below")
    return fig


def grid(const):
    points = const.matrix.ravel()
    xs, ys = points.real, points.imag
    edge = np.abs(np.append(xs, ys)).max() + 1
    fig, ax = plane((-edge, edge), (-edge, edge), 32)
    ticks = np.arange(1, edge)
    xticks(ax, np.append(-ticks, ticks))
    yticks(ax, np.append(-ticks, ticks))
    # Dashed lines through rows and columns
    for x in np.unique(xs):
        column = ys[xs == x]
        line(ax, (x, column.min()), (x, column.max()), dashes=DASHES)
    for y in np.unique(ys):
        row = xs[ys == y]
        line(ax, (row.min(), y), (row.max(), y), dashes=DASHES)
    dots(ax, points, ["above"] * len(points))
    # Coordinates in the margins
    for x in np.unique(xs):
        label(ax, (x, ys.min() - 0.25), x, "below")
    for y in np.unique(ys):
        label(ax, (xs.min() - 0.25, y), y, "left")
    return fig
