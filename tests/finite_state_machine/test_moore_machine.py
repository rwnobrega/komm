from re import escape

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
    np.testing.assert_equal(output, expected_output)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"transitions": [0, 1]}, "'transitions' must be a 2D-array"),
        ({"transitions": [[0, 2], [1, 0]]}, "'transitions' must be in [0:2)"),
        ({"transitions": [[0, -1], [1, 0]]}, "'transitions' must be in [0:2)"),
        ({"outputs": [0, -1]}, "'outputs' must be at least 0"),
        ({"outputs": [0, 1, 1]}, "'outputs' must have shape (2,) (got (3,))"),
    ],
)
def test_moore_machine_invalid(kwargs, message):
    kwargs = {"transitions": [[0, 1], [1, 0]], "outputs": [0, 1]} | kwargs
    with pytest.raises(ValueError, match=escape(message)):
        komm.MooreMachine(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"transitions": [[0, 1.5], [1, 0]]}, "'transitions' must contain only"),
        ({"outputs": [0, 1.0]}, "'outputs' must contain only"),
    ],
)
def test_moore_machine_not_integer(kwargs, message):
    kwargs = {"transitions": [[0, 1], [1, 0]], "outputs": [0, 1]} | kwargs
    with pytest.raises(TypeError, match=message):
        komm.MooreMachine(**kwargs)
