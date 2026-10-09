from re import escape

import numpy as np
import pytest

import komm


@pytest.mark.parametrize(
    "transition_matrix, message",
    [
        ([0.5, 0.5], "'transition_matrix' must be a 2D-array"),
        ([[0.5, 0.6], [0.5, 0.5]], "rows of 'transition_matrix' must sum to 1.0"),
        ([[-0.5, 1.5], [0.5, 0.5]], "'transition_matrix' must be non-negative"),
    ],
)
def test_dmf_invalid_transition_matrix(transition_matrix, message):
    with pytest.raises(ValueError, match=message):
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
        (  # BSC plus a mixture of its rows
            [[0.9, 0.1], [0.1, 0.9], [0.8992, 0.1008]],
            1 - komm.binary_entropy(0.1),
        ),
    ],
)
def test_channel_capacity(transition_matrix, expected):
    channel = komm.DiscreteMemorylessChannel(transition_matrix)
    assert np.allclose(channel.capacity(), expected)


def test_dmc_mutual_information_size():
    dmc = komm.DiscreteMemorylessChannel([[0.6, 0.3, 0.1], [0.7, 0.1, 0.2]])
    message = "'input_pmf' must have size 2 (got 3)"
    with pytest.raises(ValueError, match=escape(message)):
        dmc.mutual_information([0.2, 0.3, 0.5])
