import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()  # Ignore the user's matplotlibrc
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,  # Asked by cmr10
    "font.size": 14.4,  # \Large, as in the Ipe figures
    "mathtext.fontset": "cm",
    "lines.scale_dashes": False,
    "lines.solid_capstyle": "butt",
    "svg.fonttype": "path",
    "svg.hashsalt": "komm",
})

THIN = 0.4
THICK = 2.0
DASHES = (4, 4)
GAP = 7

SIDES = {
    "above": ((0, GAP), "center", "bottom"),
    "below": ((0, -GAP), "center", "top"),
    "left": ((-GAP, 0), "right", "center"),
    "right": ((GAP, 0), "left", "center"),
    "above left": ((-GAP / 2, 0), "right", "bottom"),
    "above right": ((GAP / 2, GAP / 2), "left", "bottom"),
    "below right": ((GAP / 2, -GAP), "left", "top"),
}


def canvas(xlim, ylim, scale, xaxis=True, yaxis=True):
    # Scale in points per unit
    size = (np.ptp(xlim) * scale[0] / 72, np.ptp(ylim) * scale[1] / 72)
    fig = plt.figure(figsize=size)
    ax = fig.add_axes((0, 0, 1, 1), xlim=xlim, ylim=ylim)
    ax.set_axis_off()
    if xaxis:
        arrow(ax, (xlim[0], 0), (xlim[1], 0))
    if yaxis:
        arrow(ax, (0, ylim[0]), (0, ylim[1]))
    return fig, ax


def arrow(ax, start, end):
    style = "-|>,head_length=7,head_width=2.33"
    props = dict(arrowstyle=style, mutation_scale=1, shrinkA=0, shrinkB=0)
    ax.annotate("", end, start, arrowprops=props | dict(lw=THIN, color="black"))


def line(ax, start, end, **kwargs):
    ax.plot(*zip(start, end), color="black", lw=THIN, **kwargs)


def xticks(ax, xs):
    ax.plot(xs, np.zeros_like(xs), "|", color="black", ms=8, mew=THIN)


def yticks(ax, ys):
    ax.plot(np.zeros_like(ys), ys, "_", color="black", ms=8, mew=THIN)


def text(ax, xy, string, side):
    # Side as a name or as (offset, ha, va)
    offset, ha, va = SIDES[side] if isinstance(side, str) else side
    ax.annotate(string, xy, offset, textcoords="offset points", ha=ha, va=va)


def label(ax, xy, value, side):
    # Minus sign hangs when centered
    hang = r"\phantom{-}" if value < 0 and side in ["above", "below"] else ""
    text(ax, xy, f"${value:g}{hang}$", side)
