import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (
            [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, +1, 0, 0, -1, +1, -1, 0, +1, 0, 0, -1, 0, 0, +1],
        ),
        (
            [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            [+1, -1, 0, 0, +1, 0, 0, -1, +1, -1, +1, 0, 0, 0, -1],
        ),
    ],
)
def test_finite_state_machine_ami(input, expected_output):
    expected_output = np.mod(expected_output, 3)  # -1 ≡ 2

    # Mealy machine:
    # states: 0 -> last bit 1 was -1
    #         1 -> last bit 1 was +1
    mealy_machine = komm.MealyMachine(
        transitions=[[0, 1], [1, 0]],
        outputs=[[0, 1], [0, 2]],
    )
    output, _ = mealy_machine.process(input, 0)
    np.testing.assert_equal(output, expected_output)

    # Moore machine:
    # states: 0 -> output  0, last bit 1 was -1
    #         1 -> output +1, last bit 1 was +1
    #         2 -> output  0, last bit 1 was +1
    #         3 -> output -1, last bit 1 was -1
    moore_machine = komm.MooreMachine(
        transitions=[[0, 1], [2, 3], [2, 3], [0, 1]],
        outputs=[0, 1, 0, 2],
    )
    output, _ = moore_machine.process(input, 0)
    np.testing.assert_equal(output, expected_output)


@pytest.mark.parametrize(
    "machine",
    [
        komm.MealyMachine(
            transitions=[[0, 1], [1, 0]],
            outputs=[[0, 1], [0, 2]],
        ),
        komm.MooreMachine(
            transitions=[[0, 1], [2, 3], [2, 3], [0, 1]],
            outputs=[0, 1, 0, 2],
        ),
    ],
)
def test_finite_state_machine_process_invalid_input(machine):
    with pytest.raises(ValueError, match="'input' must be a 1D-array"):
        machine.process([[0, 1], [1, 0]], 0)
    with pytest.raises(ValueError, match="elements of 'input' must be in"):
        machine.process([0, 2], 0)
    with pytest.raises(ValueError, match="elements of 'input' must be in"):
        machine.process([0, -1], 0)
    with pytest.raises(TypeError, match="'input' must contain only integers"):
        machine.process([0.0, 1.0], 0)
    with pytest.raises(ValueError, match="'initial_state' must be in"):
        machine.process([0, 1], -1)
    with pytest.raises(TypeError, match="'initial_state' must be an integer"):
        machine.process([0, 1], 1.0)
