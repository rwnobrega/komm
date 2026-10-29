import matplotlib.pyplot as plt
import numpy as np

import komm

STYLE = "default"

LENGTH = 300


def sequences():
    # As in the gallery page
    komm.global_rng.set(np.random.default_rng(seed=42))
    source = komm.MarkovSource(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        pmfs=[[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]],
    )
    x = source.emit(100_000)
    z = komm.DiscreteMemorylessSource(2).emit(100_000)
    return x[:LENGTH], z[:LENGTH]


def phrases(tokens):
    words = [()]
    for p, x in tokens:
        words.append(words[p] + ((x,) if x >= 0 else ()))
    return set(words)


def layout(words):
    # Leaves in order, parents centered
    pos = {}
    leaves = 0

    def visit(word):
        nonlocal leaves
        children = [word + (x,) for x in (0, 1) if word + (x,) in words]
        if not children:
            pos[word] = (leaves, -len(word))
            leaves += 1
            return
        for child in children:
            visit(child)
        xs = [pos[child][0] for child in children]
        pos[word] = ((min(xs) + max(xs)) / 2, -len(word))

    visit(())
    return pos


def tree(ax, words, title):
    pos = layout(words)
    for word in words - {()}:
        (x0, y0), (x1, y1) = pos[word[:-1]], pos[word]
        color = "C0" if word[-1] == 0 else "C3"
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.2)
    ax.plot(*zip(*pos.values()), "o", color="black", ms=3)
    ax.set_title(title)
    ax.set_axis_off()


def plot():
    lz78 = komm.LempelZiv78Code(2)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    for ax, seq, title in zip(axs, sequences(), ["Markov source", "Fair coin"]):
        tree(ax, phrases(lz78.source_to_tokens(seq)), title)
    return fig
