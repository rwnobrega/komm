import komm

from ._machines import markov


def plot():
    chain = komm.MarkovChain([
        [1 / 2, 1 / 4, 1 / 4],
        [1 / 2, 0, 1 / 2],
        [1 / 4, 1 / 4, 1 / 2],
    ])
    return markov(chain, [(-96, 0), (0, 0), (96, 0)])
