import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()  # Ignore the user's matplotlibrc
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "axes.formatter.use_mathtext": True,  # Asked by cmr10
    "font.size": 10,
    "mathtext.fontset": "cm",
    "lines.scale_dashes": False,
    "lines.solid_capstyle": "butt",
    "svg.fonttype": "path",
    "svg.hashsalt": "komm",
})

SCALE = 32  # points per unit
THIN = 0.4
THICK = 2.0
DASHES = (4, 4)
GAP = 7

SIDES = {
    "above": ((0, GAP), "center", "bottom"),
    "below": ((0, -GAP), "center", "top"),
    "left": ((-GAP, 0), "right", "center"),
    "right": ((GAP, 0), "left", "center"),
}


def canvas(xlim, ylim):
    size = (np.ptp(xlim) * SCALE / 72, np.ptp(ylim) * SCALE / 72)
    fig = plt.figure(figsize=size)
    ax = fig.add_axes((0, 0, 1, 1), xlim=xlim, ylim=ylim)
    ax.set_axis_off()
    arrow(ax, (xlim[0], 0), (xlim[1], 0))
    arrow(ax, (0, ylim[0]), (0, ylim[1]))
    return fig, ax


def arrow(ax, start, end):
    style = "-|>,head_length=7,head_width=2.33"
    props = dict(arrowstyle=style, mutation_scale=1, shrinkA=0, shrinkB=0)
    ax.annotate("", end, start, arrowprops=props | dict(lw=THIN, color="black"))


def line(ax, start, end, **kwargs):
    ax.plot(*zip(start, end), color="black", lw=THIN, **kwargs)


def label(ax, xy, value, side):
    offset, ha, va = SIDES[side]
    # Minus sign hangs when centered
    hang = r"\phantom{-}" if value < 0 and ha == "center" else ""
    text = f"${value:g}{hang}$"
    ax.annotate(text, xy, offset, textcoords="offset points", ha=ha, va=va)
