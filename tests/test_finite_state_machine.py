import pytest

import numpy as np
import komm


def test_fsm_viterbi():
    # Sklar.01, p. 401-405.
    def metric_function(y, z):
        s = komm.int2binlist(y, width=len(z))
        return np.count_nonzero(z != s)
    fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
    z = np.array([(1, 1), (0, 1), (0, 1), (1, 0), (0, 1)])
    initial_metrics = [0.0, np.inf, np.inf, np.inf]
    input_sequences_hat, final_metrics = fsm.viterbi(z, metric_function, initial_metrics)
    assert np.allclose(final_metrics, [2.0, 2.0, 2.0, 1.0])
    assert np.array_equal(input_sequences_hat.T, [[1,1,0,0,0], [1,1,0,0,1], [1,1,1,1,0], [1,1,0,1,1]])

    # Ryan.Lin.09, p. 176-177
    def metric_function(y, z):
        y = (-1)**komm.int2binlist(y, width=len(z))
        return -np.dot(z, y)
    fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
    z = np.array([(-0.7, -0.5), (-0.8, -0.6), (-1.1, +0.4), (+0.9, +0.8)])
    initial_metrics = [0.0, np.inf, np.inf, np.inf]
    input_sequences_hat, final_metrics = fsm.viterbi(z, metric_function, initial_metrics)
    assert np.allclose(final_metrics, [-3.8, -3.4, -2.6, -2.4])
    assert np.array_equal(input_sequences_hat.T, [[1,0,0,0], [0,1,0,1], [1,1,1,0], [1,1,1,1]])


def test_fsm_forward_backward():
    # Lin.Costello.04, p. 572-575.
    fsm = komm.FiniteStateMachine(next_states=[[0,1], [1,0]], outputs=[[0,3], [2,1]])
    input_posteriors = fsm.forward_backward(
        observed_sequence=-np.array([(0.8, 0.1), (1.0, -0.5), (-1.8, 1.1), (1.6, -1.6)]),
        metric_function=lambda y, z: 0.5 * np.dot(z, (-1)**komm.int2binlist(y, width=len(z))),
        initial_state_distribution=[1, 0],
        final_state_distribution=[1, 0])
    with np.errstate(divide='ignore'):
        llr = np.log(input_posteriors[:, 0] / input_posteriors[:, 1])
    assert np.allclose(-llr, [0.48, 0.62, -1.02, 2.08], atol=0.05)

    # Abrantes.10, p.434-437
    fsm = komm.FiniteStateMachine(next_states=[[0,2], [0,2], [1,3], [1,3]], outputs=[[0,3], [3,0], [1,2], [2,1]])
    input_posteriors = fsm.forward_backward(
        observed_sequence=-np.array([(0.3, 0.1), (-0.5, 0.2), (0.8, 0.5), (-0.5, 0.3), (0.1, -0.7), (1.5, -0.4)]),
        metric_function=lambda y, z: 2.5 * np.dot(z, (-1)**komm.int2binlist(y, width=len(z))),
        initial_state_distribution=[1, 0, 0, 0],
        final_state_distribution=[1, 0, 0, 0])
    with np.errstate(divide='ignore'):
        llr = np.log(input_posteriors[:,0] / input_posteriors[:,1])
    assert np.allclose(-llr, [1.78, 0.24, -1.97, 5.52, -np.inf, -np.inf], atol=0.05)
