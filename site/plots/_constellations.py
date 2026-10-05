import numpy as np
from matplotlib.patches import Arc, Circle

from . import DASHES, GAP, THIN, canvas, label, line, text, xticks, yticks


def plane(xlim, ylim, scale):
    fig, ax = canvas(xlim, ylim, scale=(scale, scale))
    text(ax, (xlim[1], 0), "Re", "right")
    text(ax, (0, ylim[1]), "Im", "above")
    return fig, ax


def dots(ax, points, sides, symbol="x"):
    ax.plot(points.real, points.imag, "o", color="black", ms=6, mew=0)
    for i, (z, side) in enumerate(zip(points, sides)):
        text(ax, (z.real, z.imag), f"${symbol}_{{{i}}}$", side)


def outward(z, flip=False):
    u = z / abs(z)
    # Off the axes, diagonally
    if np.isclose(u.imag, 0):
        u += -1j if flip else 1j
    elif np.isclose(u.real, 0):
        u += 1 if flip else -1
    u /= abs(u)
    ha = "left" if u.real > 0.3 else "right" if u.real < -0.3 else "center"
    va = "bottom" if u.imag > 0.3 else "top" if u.imag < -0.3 else "center"
    return (GAP * u.real, GAP * u.imag), ha, va


def angle(ax, z, radius, string):
    line(ax, (0, 0), (z.real, z.imag), dashes=DASHES)
    theta = np.angle(z, deg=True)
    ax.add_patch(Arc((0, 0), 2 * radius, 2 * radius, theta2=theta, lw=THIN))
    w = radius * np.exp(1j * np.angle(z) / 2)
    text(ax, (w.real, w.imag), string, outward(w))


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


def rings(const, ticks):
    points = const.matrix.ravel()
    radii = np.unique(np.abs(points).round(9))
    edge = 4 / 3 * radii.max()
    fig, ax = plane((-edge, edge), (-edge, edge), 96 / radii.max())
    for radius in radii:
        circle = Circle((0, 0), radius, fill=False, lw=THIN, ls=(0, DASHES))
        ax.add_patch(circle)
    xticks(ax, np.append(np.negative(ticks), ticks))
    yticks(ax, np.append(np.negative(ticks), ticks))
    dots(ax, points, [outward(z) for z in points])
    for radius in np.intersect1d(radii, ticks):
        for z in radius * np.array([1, 1j, -1, -1j]):
            value = z.real + z.imag
            text(ax, (z.real, z.imag), f"${value:g}$", outward(z, flip=True))
    return fig, ax
