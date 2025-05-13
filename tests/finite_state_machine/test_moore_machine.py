import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([1, 0, 1], [0, 1, 0]),
        ([1, 1, 1, 1, 1], [0, 1, 1, 1, 1]),
        ([0, 1, 0, 0, 1], [0, 0, 1, 0, 0]),
        ([1, 0, 0], [0, 1, 0]),
        ([1, 1, 0], [0, 1, 1]),
        ([1], [0]),
        ([0], [0]),
    ],
)
def test_moore_machine_halve(input, expected_output):
    # From [https://jflap.org/tutorial/mealy/mooreExamples.html].
    machine = komm.MooreMachine(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        outputs=[0, 0, 1, 1],
    )
    output, _ = machine.process(input, 0)
    np.testing.assert_array_equal(output, expected_output)
