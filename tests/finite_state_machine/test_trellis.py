import numpy as np

from komm._finite_state_machine.trellis import TrellisSection


def test_trellis_section_incoming():
    # Parallel, missing, and unreachable cases
    section = TrellisSection([[0, 0], [0, -1]], [[3, 2], [1, 0]], 2)
    in_states, in_inputs, in_outputs = section.incoming
    np.testing.assert_equal(in_states, [[0, 0, 1], [-1, -1, -1]])
    np.testing.assert_equal(in_inputs[0], [0, 1, 0])
    np.testing.assert_equal(in_outputs[0], [3, 2, 1])
