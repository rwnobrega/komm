import numpy as np

from . import RADIUS, THICK, THIN, arrow, canvas, line, text

# Coordinates in radii
STEP = 4
PORT = 1 / 3


def lfsr(sequence):
    exponents = sequence.feedback_polynomial.exponents()
    n = exponents[-1]
    # Bit i at (STEP * i, 0); adder above
    hub = (1, 5)
    xlim = (-1 - THIN / 2 / RADIUS, STEP * n + 2)
    ylim = (-1 - THICK / 2 / RADIUS, hub[1] + 1 + THICK / 2 / RADIUS)
    fig, ax = canvas(xlim, ylim, scale=(RADIUS, RADIUS), xaxis=False, yaxis=False)
    adder(ax, hub)
    wire(ax, (0, 5), (-1, 5), (-1, 0), (1, 0))
    chain(ax, n)
    arrow(ax, (STEP * n - 1, 0), (STEP * n + 2, 0))
    # One lane per tap, the last on top
    for lane, i in enumerate(reversed(exponents[1:])):
        y = hub[1] - lane
        dot(ax, (STEP * i, 0))
        wire(ax, (STEP * i, 0), (STEP * i, y), (3, y), hub, shrinkB=RADIUS)
    for i in range(n + 1):
        bit(ax, STEP * i, i)
    return fig


def low_rate(code):
    m = code.degree
    xlim = (-3, STEP * m + 3)
    ylim = (-6 - THIN / 2 / RADIUS, 6 + THIN / 2 / RADIUS)
    fig, ax = canvas(xlim, ylim, scale=(RADIUS, RADIUS), xaxis=False, yaxis=False)
    port(ax, (xlim[0], 0))
    arrow(ax, (xlim[0], 0), (1, 0))
    chain(ax, m)
    line(ax, (STEP * m - 1, 0), (STEP * m + 1, 0))
    # End joints step aside, for the labels
    joints = STEP * np.arange(m + 1)
    joints[[0, -1]] += [-1, 1]
    taps = [g.exponents() for g in code.g_row]
    # One adder above, one below
    for exponents, y in zip(taps, [4, -4]):
        hub = (2 * m, y)
        adder(ax, hub)
        feed(ax, joints[exponents], hub)
        s = np.sign(y)
        line(ax, (2 * m, y + s), (2 * m, y + 2 * s), (xlim[1] - PORT, y + 2 * s))
        port(ax, (xlim[1], y + 2 * s))
    for i in np.unique(np.concatenate(taps)):
        dot(ax, (joints[i], 0))
    for i in range(m + 1):
        # Clear of the taps going down
        shift = 0.5 if 0 < i < m and i in taps[1] else 0
        bit(ax, STEP * i + shift, i)
    return fig


def chain(ax, n):
    for k in range(1, n + 1):
        register(ax, (STEP * k - 2, 0))
    for k in range(1, n):
        arrow(ax, (STEP * k - 1, 0), (STEP * k + 1, 0))


def feed(ax, xs, hub):
    # Outermost taps reach the sides
    hx, hy = hub
    s = np.sign(hy)
    left, right = xs[xs < hx], xs[xs > hx]
    for x in xs:
        if x == hx:
            wire(ax, (x, 0), (x, hy - s))
        elif x in (left.min(initial=hx), right.max(initial=hx)):
            wire(ax, (x, 0), (x, hy - s), (hx + np.sign(x - hx), hy))
        else:
            wire(ax, (x, 0), (x, hy - 2 * s), hub, shrinkB=RADIUS)


def bit(ax, x, i):
    # Digits centered, as in the nodes
    text(ax, (x, -1), f"${i}$", ((0, 0), "center", "center_baseline"))


def register(ax, xy):
    ax.plot(*xy, "s", color="black", ms=2 * RADIUS, mfc="none", mew=THICK)


def adder(ax, xy):
    # Circled plus, as large as a node
    for marker, size in [("o", 2 * RADIUS), ("+", RADIUS)]:
        ax.plot(*xy, marker, color="black", ms=size, mfc="none", mew=THICK)


def dot(ax, xy):
    ax.plot(*xy, "o", color="black", ms=3.6, mew=0)


def port(ax, tip):
    # Hollow triangle pointing right
    props = dict(color="black", ms=PORT * RADIUS, mfc="none", mew=0.8, clip_on=False)
    ax.plot(tip[0] - PORT / 2, tip[1], ">", **props)


def wire(ax, *points, **kwargs):
    # Arrow on the last stretch only
    if len(points) > 2:
        line(ax, *points[:-1])
    arrow(ax, points[-2], points[-1], **kwargs)
