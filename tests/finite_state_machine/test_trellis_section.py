from re import escape

import numpy as np
import pytest

import komm


def test_trellis_section_incoming():
    # Parallel, missing, and unreachable cases
    section = komm.TrellisSection([[0, 0], [0, -1]], [[3, 2], [1, 0]], 2)
    in_states, in_inputs, in_outputs = section.incoming
    np.testing.assert_equal(in_states, [[0, 0, 1], [-1, -1, -1]])
    np.testing.assert_equal(in_inputs[0], [0, 1, 0])
    np.testing.assert_equal(in_outputs[0], [3, 2, 1])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"transitions": [0, -1]}, "'transitions' must be a 2D-array"),
        ({"transitions": [[0, 2], [-1, 0]]}, "'transitions' must be in [-1:2)"),
        ({"transitions": [[0, -2], [-1, 0]]}, "'transitions' must be in [-1:2)"),
        ({"outputs": [[0, -1], [0, 1]]}, "'outputs' must be at least 0"),
        ({"outputs": [0, 1]}, "'outputs' must have shape (2, 2) (got (2,))"),
        ({"num_next_states": 0}, "'num_next_states' must be a positive integer"),
    ],
)
def test_trellis_section_invalid(kwargs, message):
    kwargs = {
        "transitions": [[0, 1], [-1, 0]],
        "outputs": [[0, 1], [0, 1]],
        "num_next_states": 2,
    } | kwargs
    with pytest.raises(ValueError, match=escape(message)):
        komm.TrellisSection(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"transitions": [[0, 1.5], [-1, 0]]}, "'transitions' must contain only"),
        ({"outputs": [[0, 1.0], [0, 1]]}, "'outputs' must contain only"),
        ({"num_next_states": 2.0}, "'num_next_states' must be an integer"),
    ],
)
def test_trellis_section_not_integer(kwargs, message):
    kwargs = {
        "transitions": [[0, 1], [-1, 0]],
        "outputs": [[0, 1], [0, 1]],
        "num_next_states": 2,
    } | kwargs
    with pytest.raises(TypeError, match=message):
        komm.TrellisSection(**kwargs)


def test_trellis_section_repr():
    section = komm.TrellisSection([[0, -1]], [[0, 1]], 1)
    args = "transitions=[[0, -1]], outputs=[[0, 1]], num_next_states=1"
    assert repr(section) == f"TrellisSection({args})"
