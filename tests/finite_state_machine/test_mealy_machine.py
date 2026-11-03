from re import escape

import numpy as np
import pytest

import komm


def test_mealy_machine_parallel_transitions():
    # Two inputs lead to each next state
    machine = komm.MealyMachine(
        transitions=[[0, 0, 1, 1], [0, 0, 1, 1]],
        outputs=[[0, 1, 2, 3], [3, 2, 1, 0]],
    )
    input = [0, 2, 1, 3, 0, 0, 2]
    output, _ = machine.process(input, 0)
    metrics = np.array([0.0, np.inf])
    input_hat = machine.viterbi_streaming(
        observed=output,
        metric_function=lambda y, z: float(y != z),
        memory={"paths": np.zeros((2, 4), dtype=int), "metrics": metrics},
    )
    np.testing.assert_equal(input_hat[3:], input[:-3])  # Delay of 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"transitions": [0, 1]}, "'transitions' must be a 2D-array"),
        ({"transitions": [[0, 2], [1, 0]]}, "'transitions' must be in [0:2)"),
        ({"transitions": [[0, -1], [1, 0]]}, "'transitions' must be in [0:2)"),
        ({"outputs": [[0, -1], [0, 2]]}, "'outputs' must be at least 0"),
        ({"outputs": [0, 1]}, "'outputs' must have shape (2, 2) (got (2,))"),
    ],
)
def test_mealy_machine_invalid(kwargs, message):
    kwargs = {"transitions": [[0, 1], [1, 0]], "outputs": [[0, 1], [0, 2]]} | kwargs
    with pytest.raises(ValueError, match=escape(message)):
        komm.MealyMachine(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"transitions": [[0, 1.5], [1, 0]]}, "'transitions' must contain only"),
        ({"outputs": [[0, 1.0], [0, 2]]}, "'outputs' must contain only"),
    ],
)
def test_mealy_machine_not_integer(kwargs, message):
    kwargs = {"transitions": [[0, 1], [1, 0]], "outputs": [[0, 1], [0, 2]]} | kwargs
    with pytest.raises(TypeError, match=message):
        komm.MealyMachine(**kwargs)
