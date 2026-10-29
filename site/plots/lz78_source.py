import komm

from ._machines import markov_source


def plot():
    source = komm.MarkovSource(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        pmfs=[[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]],
    )
    return markov_source(source, [(-96, 0), (0, 64), (0, -64), (96, 0)])
