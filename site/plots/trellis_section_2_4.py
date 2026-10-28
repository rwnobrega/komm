import komm

from ._trellises import diagram


def plot():
    section = komm.TrellisSection(
        transitions=[[0, 1], [2, 3]],
        outputs=[[0, 3], [1, 2]],
    )
    trellis = komm.Trellis([section])
    return diagram(trellis)
