import komm

from ._machines import moore


def plot():
    machine = komm.MooreMachine(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        outputs=[0, 0, 1, 1],
    )
    return moore(machine, [(-96, 0), (0, 64), (0, -64), (96, 0)])
