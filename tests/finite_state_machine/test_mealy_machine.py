from re import escape

import pytest

import komm


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
