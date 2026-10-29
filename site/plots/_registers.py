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
    line(ax, (STEP * n - 1, 0), (xlim[1] - PORT, 0))
    port(ax, (xlim[1], 0))
    # One lane per tap, the last on top
    for lane, i in enumerate(reversed(exponents[1:])):
        y = hub[1] - lane
        dot(ax, (STEP * i, 0))
        wire(ax, (STEP * i, 0), (STEP * i, y), (3, y), hub, shrinkB=RADIUS)
    for i in range(n + 1):
        bit(ax, (STEP * i, 0), i)
    return fig


def convolutional(code):
    return encoder(code.feedforward_polynomials)


def low_rate(code):
    return encoder([code.g_row])


def encoder(polys):
    # Chain i at y = -8i; adder j at 4 - 8j
    m = max(p.degree for row in polys for p in row)
    hubs = [(2 * m, 4 - 8 * j) for j in range(len(polys[0]))]
    # Outputs two radii past the outer adders
    top, bottom = hubs[0][1] + 2, hubs[-1][1] - 2
    xlim = (-3, STEP * m + 3)
    # Room for the ports
    ylim = (bottom - PORT / 2, top + PORT / 2)
    fig, ax = canvas(xlim, ylim, scale=(RADIUS, RADIUS), xaxis=False, yaxis=False)
    for i, row in enumerate(polys):
        y = -8 * i
        port(ax, (xlim[0], y))
        arrow(ax, (xlim[0], y), (1, y))
        tapped_chain(ax, y, [p.exponents() for p in row], hubs)
    for j, (hx, hy) in enumerate(hubs):
        adder(ax, (hx, hy))
        if 0 < j < len(hubs) - 1:
            points = [(hx + 1, hy)]
        else:
            # Outer adders: out, then right
            s = 1 if j == 0 else -1
            points = [(hx, hy + s), (hx, hy + 2 * s)]
        y = points[-1][1]
        line(ax, *points, (xlim[1] - PORT, y))
        port(ax, (xlim[1], y))
    return fig


def feedback(code):
    # One input, through the feedback adder
    (row,) = code.feedforward_polynomials
    (q,) = code.feedback_polynomials
    (parity,) = [p for p in row if p != q]
    m = max(q.degree, parity.degree)
    # Parity adder above, feedback adder below
    hubs = [(2 * m, 4), (2 * m, -4)]
    xlim = (-9, STEP * m + 3)
    # Room for the lower adder and ports
    ylim = (-5 - THICK / 2 / RADIUS, 8 + PORT / 2)
    fig, ax = canvas(xlim, ylim, scale=(RADIUS, RADIUS), xaxis=False, yaxis=False)
    # Input, branch at -7, adder at -4
    port(ax, (xlim[0], 0))
    arrow(ax, (xlim[0], 0), (-5, 0))
    dot(ax, (-7, 0))
    adder(ax, (-4, 0))
    arrow(ax, (-3, 0), (1, 0))
    # Feedback taps skip bit 0
    tapped_chain(ax, 0, [parity.exponents(), q.exponents()[1:]], hubs)
    for hub in hubs:
        adder(ax, hub)
    wire(ax, (2 * m - 1, -4), (-4, -4), (-4, -1))
    line(ax, (2 * m, 5), (2 * m, 6), (xlim[1] - PORT, 6))
    port(ax, (xlim[1], 6))
    # Systematic output, from the branch
    line(ax, (-7, 0), (-7, 8), (xlim[1] - PORT, 8))
    port(ax, (xlim[1], 8))
    return fig


def high_rate(code):
    # Observer form: adders between the registers
    *taps, q = [p.exponents() for p in code.h_row]
    m = code.degree
    # Input j at y = 2(k - j) + 2, above
    lanes = 2 * (len(taps) - np.arange(len(taps))) + 2
    xlim = (-2, 2 * STEP * m + 5)
    # Room for the ports and dots
    ylim = (-4 - PORT / 2, lanes[0] + PORT / 2)
    fig, ax = canvas(xlim, ylim, scale=(RADIUS, RADIUS), xaxis=False, yaxis=False)
    for y in lanes:
        port(ax, (xlim[0], y))
        line(ax, (xlim[0], y), (xlim[1] - PORT, y))
        port(ax, (xlim[1], y))
    # Adder of D^i at x = 2 STEP (m - i)
    for i in range(m + 1):
        x = 2 * STEP * (m - i)
        adder(ax, (x, 0))
        text(ax, (x + 1.5, 1), f"${i}$", ((0, 0), "center", "center_baseline"))
        if i > 0:
            arrow(ax, (x + 1, 0), (x + 3, 0))
            register(ax, (x + 4, 0))
            arrow(ax, (x + 5, 0), (x + 7, 0))
        # Taps side by side, 6 pt apart
        js = [j for j, e in enumerate(taps) if i in e]
        for j, dx in zip(js, (np.arange(len(js)) - (len(js) - 1) / 2) * 6 / RADIUS):
            dot(ax, (x + dx, lanes[j]))
            arrow(ax, (x + dx, lanes[j]), (x + dx, 1))
    end = 2 * STEP * m
    line(ax, (end + 1, 0), (xlim[1] - PORT, 0))
    port(ax, (xlim[1], 0))
    # Feedback from the output, below
    xs = 2 * STEP * (m - q[q > 0])
    if xs.size > 0:
        dot(ax, (end + 3, 0))
        wire(ax, (end + 3, 0), (end + 3, -4), (xs.min(), -4), (xs.min(), -1))
        for x in xs[xs > xs.min()]:
            dot(ax, (x, -4))
            arrow(ax, (x, -4), (x, -1))
    return fig


def tapped_chain(ax, y, taps, hubs):
    degree = np.concatenate(taps).max()
    chain(ax, degree, y)
    line(ax, (STEP * degree - 1, y), (STEP * degree + 1, y))
    # End joints step aside, for the labels
    joints = STEP * np.arange(degree + 1)
    joints[[0, -1]] += [-1, 1]
    for exponents, hub in zip(taps, hubs):
        feed(ax, joints[exponents], y, hub)
    for b in np.unique(np.concatenate(taps)):
        dot(ax, (joints[b], y))
    # Clear of the taps going down
    down = {b for e, (_, hy) in zip(taps, hubs) if hy < y for b in e}
    for b in range(degree + 1):
        shift = 0.5 if 0 < b < degree and b in down else 0
        bit(ax, (STEP * b + shift, y), b)


def chain(ax, n, y=0):
    for k in range(1, n + 1):
        register(ax, (STEP * k - 2, y))
    for k in range(1, n):
        arrow(ax, (STEP * k - 1, y), (STEP * k + 1, y))


def feed(ax, xs, y, hub):
    # Far taps enter by the side
    hx, hy = hub
    s = np.sign(hy - y)
    for x in xs:
        if x == hx:
            wire(ax, (x, y), (x, hy - s))
        elif abs(x - hx) > 3 * STEP:
            wire(ax, (x, y), (x, hy - s), (hx + np.sign(x - hx), hy))
        else:
            wire(ax, (x, y), (x, hy - 2 * s), hub, shrinkB=RADIUS)


def bit(ax, xy, i):
    # Digits centered, as in the nodes
    x, y = xy
    text(ax, (x, y - 1), f"${i}$", ((0, 0), "center", "center_baseline"))


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
