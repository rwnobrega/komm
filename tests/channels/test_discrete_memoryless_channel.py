import numpy as np
import pytest

import komm
import komm.abc


@pytest.mark.parametrize(
    "transition_matrix",
    [
        [0.5, 0.5],  # Not a 2D array
        [[0.5, 0.6], [0.5, 0.5]],  # Does not sum to 1.0
        [[-0.5, 1.5], [0.5, 0.5]],  # Negative probability
    ],
)
def test_dmf_invalid_transition_matrix(transition_matrix):
    with pytest.raises(ValueError):
        komm.DiscreteMemorylessChannel(transition_matrix)


def _get_noisy_typewriter_transition_matrix():
    transition_matrix = np.zeros((26, 26))
    for i in range(26):
        transition_matrix[i, i] = 0.5
        transition_matrix[i, (i + 1) % 26] = 0.5
    return transition_matrix


@pytest.mark.parametrize(
    "transition_matrix, expected",
    [
        (  # [CT06, Sec. 7.1.1]
            [[1, 0], [0, 1]],
            1.0,
        ),
        (  # [CT06, Sec. 7.1.2]
            [[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 3, 2 / 3]],
            1.0,
        ),
        (  # [CT06, Sec. 7.1.3]
            _get_noisy_typewriter_transition_matrix(),
            np.log2(13),
        ),
        (  # [CT06, Sec. 7.1.4]
            [[0.7, 0.3], [0.3, 0.7]],
            1 - komm.binary_entropy(0.3),
        ),
        (  # [CT06, Sec. 7.1.5]
            [[0.7, 0.3, 0], [0, 0.3, 0.7]],
            1 - 0.3,
        ),
        (  # [CT06, Sec. 7.2]
            [[0.3, 0.2, 0.5], [0.5, 0.3, 0.2], [0.2, 0.5, 0.3]],
            np.log2(3) - komm.entropy([0.5, 0.3, 0.2]),
        ),
        (  # [CT06, Exercise 7.8]
            [[1, 0], [0.5, 0.5]],
            komm.binary_entropy(1 / 5) - 2 / 5,
        ),
    ],
)
def test_channel_capacity(transition_matrix, expected):
    channel = komm.DiscreteMemorylessChannel(transition_matrix)
    assert np.allclose(channel.capacity(), expected)
