import komm

from ._trellises import diagram


def plot():
    section = komm.TrellisSection(
        transitions=[[0, -1], [1, -1], [0, -1], [1, -1]],
        outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
    )
    trellis = komm.Trellis([section])
    return diagram(trellis)
