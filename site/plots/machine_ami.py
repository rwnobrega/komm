import komm

from ._machines import moore


def plot():
    machine = komm.MooreMachine(
        transitions=[[0, 1], [2, 3], [2, 3], [0, 1]],
        outputs=[0, 1, 0, 2],
    )
    # Braces avoid operator spacing
    levels = ["0", "{+}", "{-}"]
    return moore(machine, [(-64, 0), (0, 64), (64, 0), (0, -64)], levels)
