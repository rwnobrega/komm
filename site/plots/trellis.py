import komm

from ._trellises import diagram


def plot():
    trellis = komm.Trellis([
        komm.TrellisSection(
            transitions=[[0, 1]],
            outputs=[[0, 3]],
        ),
        komm.TrellisSection(
            transitions=[[0, 1], [2, 3]],
            outputs=[[0, 3], [1, 2]],
        ),
        komm.TrellisSection(
            transitions=[[0, -1], [1, -1], [0, -1], [1, -1]],
            outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
        ),
        komm.TrellisSection(
            transitions=[[0, -1], [0, -1]],
            outputs=[[0, 3], [3, 0]],
        ),
    ])
    return diagram(trellis)
