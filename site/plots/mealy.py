import komm

from ._machines import mealy


def plot():
    machine = komm.MealyMachine(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
    )
    return mealy(machine, [(-96, 0), (0, 64), (0, -64), (96, 0)])
