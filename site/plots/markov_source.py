import komm

from ._machines import markov_source


def plot():
    source = komm.MarkovSource(
        transitions=[[0, 1], [0, 0]],
        pmfs=[[0.5, 0.5], [0, 1]],
    )
    return markov_source(source, [(-48, 0), (48, 0)])
