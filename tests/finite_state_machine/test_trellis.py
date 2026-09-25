from itertools import product

import numpy as np
import pytest

import komm
from komm._finite_state_machine.trellis import TrellisSection, forward_backward, viterbi
from komm._util.special_functions import boxplus


def random_sections(rng: np.random.Generator, length: int) -> list[TrellisSection]:
    # Missing and parallel branches, one output each
    num_states = rng.integers(1, 5, size=length + 1)
    sections: list[TrellisSection] = []
    for t in range(length):
        shape = (num_states[t], 3)
        transitions = rng.integers(num_states[t + 1], size=shape)
        transitions[:, 1:][rng.random((num_states[t], 2)) < 0.3] = -1
        outputs = rng.permutation(12)[: 3 * num_states[t]].reshape(shape)
        sections.append(TrellisSection(transitions, outputs, num_states[t + 1]))
    return sections


def all_paths(sections: list[TrellisSection]):
    # Initial state, inputs, outputs, final state
    for s0 in range(sections[0].num_states):
        for inputs in product(range(sections[0].num_inputs), repeat=len(sections)):
            s, outputs = s0, []
            for section, x in zip(sections, inputs):
                if section.transitions[s, x] < 0:
                    break
                outputs.append(section.outputs[s, x])
                s = section.transitions[s, x]
            else:
                yield s0, inputs, outputs, s


def test_trellis_section_incoming():
    # Parallel, missing, and unreachable cases
    section = TrellisSection([[0, 0], [0, -1]], [[3, 2], [1, 0]], 2)
    in_states, in_inputs, in_outputs = section.incoming
    np.testing.assert_equal(in_states, [[0, 0, 1], [-1, -1, -1]])
    np.testing.assert_equal(in_inputs[0], [0, 1, 0])
    np.testing.assert_equal(in_outputs[0], [3, 2, 1])


@pytest.mark.repeat(10)
def test_trellis_viterbi_brute_force(rng):
    sections = random_sections(rng, 4)
    initial = rng.random(sections[0].num_states)
    final = rng.random(sections[-1].num_next_states)
    branch_metrics = rng.random((5, 4, 12))
    inputs_hat = viterbi(sections, branch_metrics, initial, final)
    for metrics, x_hat in zip(branch_metrics, inputs_hat):
        _, best = min(
            (initial[s0] + metrics[range(4), outputs].sum() + final[s1], inputs)
            for s0, inputs, outputs, s1 in all_paths(sections)
        )
        np.testing.assert_equal(x_hat, best)


@pytest.mark.repeat(10)
def test_trellis_viterbi_shapes(rng):
    # Leading dimensions, with broadcast end metrics
    sections = random_sections(rng, 4)
    initial = rng.random((3, sections[0].num_states))
    final = rng.random((3, sections[-1].num_next_states))
    branch_metrics = rng.random((2, 3, 4, 12))
    inputs_hat = viterbi(sections, branch_metrics, initial, final)
    assert inputs_hat.shape == (2, 3, 4)
    for i, j in product(range(2), range(3)):
        x_hat = viterbi(sections, branch_metrics[i, j], initial[j], final[j])
        np.testing.assert_equal(inputs_hat[i, j], x_hat)


@pytest.mark.repeat(10)
def test_trellis_viterbi_mealy_machine(rng):
    # Same ties as the reference implementation
    transitions = rng.integers(4, size=(4, 3))
    outputs = rng.integers(4, size=(4, 3))
    machine = komm.MealyMachine(transitions, outputs)
    observed = rng.integers(3, size=(6, 4))  # Integer costs, hence many ties
    initial = rng.integers(3, size=4).astype(float)
    inputs_hat, final_metrics = machine.viterbi(observed, lambda y, z: z[y], initial)
    section = TrellisSection(transitions, outputs, 4)
    for s in np.flatnonzero(np.isfinite(final_metrics)):
        final = np.full(4, np.inf)
        final[s] = 0.0
        inputs = viterbi([section] * 6, observed, initial, final)
        np.testing.assert_equal(inputs, inputs_hat[:, s])


@pytest.mark.repeat(10)
def test_trellis_forward_backward_brute_force(rng):
    sections = random_sections(rng, 4)
    initial = rng.normal(size=sections[0].num_states)
    final = rng.normal(size=sections[-1].num_next_states)
    branch_metrics = rng.normal(size=(5, 4, 12))
    log_posteriors = forward_backward(sections, branch_metrics, initial, final)
    for metrics, log_app in zip(branch_metrics, log_posteriors):
        expected = np.full((4, 3), -np.inf)
        for s0, inputs, outputs, s1 in all_paths(sections):
            weight = initial[s0] + metrics[range(4), outputs].sum() + final[s1]
            steps = (range(4), inputs)
            expected[steps] = np.logaddexp(expected[steps], weight)
        expected -= np.logaddexp.reduce(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(log_app, expected)


@pytest.mark.repeat(10)
def test_trellis_forward_backward_shapes(rng):
    # Leading dimensions, with broadcast end metrics
    sections = random_sections(rng, 4)
    initial = rng.normal(size=(3, sections[0].num_states))
    final = rng.normal(size=(3, sections[-1].num_next_states))
    branch_metrics = rng.normal(size=(2, 3, 4, 12))
    log_posteriors = forward_backward(sections, branch_metrics, initial, final)
    assert log_posteriors.shape == (2, 3, 4, 3)
    for i, j in product(range(2), range(3)):
        log_app = forward_backward(sections, branch_metrics[i, j], initial[j], final[j])
        np.testing.assert_allclose(log_posteriors[i, j], log_app)


def test_trellis_single_parity_check(rng):
    # Code (3, 2); state is partial parity
    sections = [
        TrellisSection([[0, 1]], [[0, 1]], 2),
        TrellisSection([[0, 1], [1, 0]], [[0, 1], [0, 1]], 2),
        TrellisSection([[0, -1], [-1, 0]], [[0, 1], [0, 1]], 1),
    ]
    r = rng.normal(size=(100, 3))
    zeros = np.zeros_like(r)
    v_hat = viterbi(sections, np.stack([zeros, r], axis=2), [0.0], [0.0])
    decoder = komm.WagnerDecoder(komm.SingleParityCheckCode(3))
    np.testing.assert_equal(v_hat, decoder.decode_to_codeword(r))
    log_app = forward_backward(sections, np.stack([zeros, -r], axis=2), [0.0], [0.0])
    extrinsic = boxplus(r[:, [1, 2, 0]], r[:, [2, 0, 1]])
    np.testing.assert_allclose(log_app[..., 0] - log_app[..., 1], r + extrinsic)
