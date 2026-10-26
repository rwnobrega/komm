from itertools import product
from re import escape

import numpy as np
import pytest

import komm
from komm._finite_state_machine.Trellis import forward_backward, viterbi
from komm._util.special_functions import boxplus


def random_sections(rng: np.random.Generator, length: int) -> list[komm.TrellisSection]:
    # Missing and parallel branches, one output each
    num_states = int(rng.integers(1, 5))
    sections: list[komm.TrellisSection] = []
    for _ in range(length):
        shape = (num_states, 3)
        transitions = rng.integers(4, size=shape)
        transitions[:, 1:][rng.random((num_states, 2)) < 0.3] = -1
        # Renumber the entered next states
        valid = transitions >= 0
        transitions[valid] = np.unique(transitions[valid], return_inverse=True)[1]
        outputs = rng.permutation(12)[: 3 * num_states].reshape(shape)
        sections.append(komm.TrellisSection(transitions, outputs))
        num_states = sections[-1].num_next_states
    return sections


def all_paths(sections: list[komm.TrellisSection]):
    # Initial state, inputs, outputs, final state
    num_inputs = sections[0].num_input_symbols
    for s0 in range(sections[0].num_states):
        for inputs in product(range(num_inputs), repeat=len(sections)):
            s, outputs = s0, []
            for section, x in zip(sections, inputs):
                if section.transitions[s, x] < 0:
                    break
                outputs.append(section.outputs[s, x])
                s = section.transitions[s, x]
            else:
                yield s0, inputs, outputs, s


# First sections of the example trellis
SECTION_0 = komm.TrellisSection([[0, 1]], [[0, 3]])
SECTION_1 = komm.TrellisSection([[0, 1], [2, 3]], [[0, 3], [1, 2]])


def test_trellis_sequence():
    trellis = komm.Trellis([SECTION_0, SECTION_1])
    assert len(trellis) == 2
    assert trellis[-1] is SECTION_1
    assert list(trellis) == [SECTION_0, SECTION_1]
    assert isinstance(trellis[1:], komm.Trellis)
    assert list(trellis[1:]) == [SECTION_1]


@pytest.mark.repeat(10)
def test_trellis_widths(rng):
    sections = random_sections(rng, 4)
    widths = [section.num_states for section in sections]
    widths.append(sections[-1].num_next_states)
    np.testing.assert_equal(komm.Trellis(sections).widths, widths)


@pytest.mark.parametrize(
    "sections, message",
    [
        ([], "'sections' must be non-empty"),
        ([SECTION_1, SECTION_0], "section 1 must have 4 states (got 1)"),
        ([SECTION_0, SECTION_1, SECTION_1], "section 2 must have 4 states (got 2)"),
    ],
)
def test_trellis_invalid(sections, message):
    with pytest.raises(ValueError, match=escape(message)):
        komm.Trellis(sections)


def test_trellis_repr():
    section = komm.TrellisSection([[0, -1]], [[0, 1]])
    assert repr(komm.Trellis([section])) == f"Trellis(sections=[{section!r}])"


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
    transitions[:, 0] = rng.permutation(4)  # Every state entered
    outputs = rng.integers(4, size=(4, 3))
    machine = komm.MealyMachine(transitions, outputs)
    observed = rng.integers(3, size=(6, 4))  # Integer costs, hence many ties
    initial = rng.integers(3, size=4).astype(float)
    inputs_hat, final_metrics = machine.viterbi(observed, lambda y, z: z[y], initial)
    section = komm.TrellisSection(transitions, outputs)
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
    sections = komm.Trellis([
        komm.TrellisSection([[0, 1]]),
        komm.TrellisSection([[0, 1], [1, 0]]),
        komm.TrellisSection([[0, -1], [-1, 0]]),
    ])
    r = rng.normal(size=(100, 3))
    zeros = np.zeros_like(r)
    v_hat = viterbi(sections, np.stack([zeros, r], axis=2), [0.0], [0.0])
    decoder = komm.WagnerDecoder(komm.SingleParityCheckCode(3))
    np.testing.assert_equal(v_hat, decoder.decode_to_codeword(r))
    log_app = forward_backward(sections, np.stack([zeros, -r], axis=2), [0.0], [0.0])
    extrinsic = boxplus(r[:, [1, 2, 0]], r[:, [2, 0, 1]])
    np.testing.assert_allclose(log_app[..., 0] - log_app[..., 1], r + extrinsic)
