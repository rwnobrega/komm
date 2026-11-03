from itertools import product
from re import escape

import numpy as np
import pytest

import komm
from komm._util.special_functions import boxplus


def random_trellis(rng: np.random.Generator, length: int) -> komm.Trellis:
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
    return komm.Trellis(sections)


def all_paths(trellis: komm.Trellis):
    # Initial state, inputs, outputs, final state
    num_inputs = trellis[0].num_input_symbols
    for s0 in range(trellis[0].num_states):
        for inputs in product(range(num_inputs), repeat=len(trellis)):
            s, outputs = s0, []
            for section, x in zip(trellis, inputs):
                if section.transitions[s, x] < 0:
                    break
                outputs.append(section.outputs[s, x])
                s = section.transitions[s, x]
            else:
                yield s0, inputs, outputs, s


def branches(trellis: komm.Trellis, s0: int, inputs) -> list[tuple[int, int]]:
    # State and input of each branch
    states = [s0]
    for section, x in zip(trellis, inputs):
        states.append(int(section.transitions[states[-1], x]))
    return list(zip(states, inputs))


def survivors(trellis: komm.Trellis, branch_metrics: np.ndarray, initial):
    # Best path into each final state
    n = trellis[-1].num_next_states
    final = np.where(np.eye(n, dtype=bool), 0.0, np.inf)
    args = (np.broadcast_to(branch_metrics, (n, *branch_metrics.shape)), initial, final)
    inputs = trellis.viterbi(*args)
    outputs = trellis.viterbi(*args, symbols="output")
    costs = branch_metrics[range(len(trellis)), outputs].sum(axis=1)
    return inputs, costs


# First sections of the example trellis
SECTION_0 = komm.TrellisSection([[0, 1]], [[0, 3]])
SECTION_1 = komm.TrellisSection([[0, 1], [2, 3]], [[0, 3], [1, 2]])

# Output bits, LSB-first
BITS = komm.int_to_bits(range(4), width=2).reshape(-1, 2)


def test_trellis_sequence():
    trellis = komm.Trellis([SECTION_0, SECTION_1])
    assert len(trellis) == 2
    assert trellis[-1] is SECTION_1
    assert list(trellis) == [SECTION_0, SECTION_1]
    assert isinstance(trellis[1:], komm.Trellis)
    assert list(trellis[1:]) == [SECTION_1]


@pytest.mark.repeat(10)
def test_trellis_widths(rng):
    trellis = random_trellis(rng, 4)
    widths = [section.num_states for section in trellis]
    widths.append(trellis[-1].num_next_states)
    np.testing.assert_equal(trellis.widths, widths)


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
    trellis = random_trellis(rng, 4)
    initial = rng.random(trellis[0].num_states)
    final = rng.random(trellis[-1].num_next_states)
    branch_metrics = rng.random((5, 4, 12))
    args = (branch_metrics, initial, final)
    inputs_hat = trellis.viterbi(*args, symbols="input")
    outputs_hat = trellis.viterbi(*args, symbols="output")
    for metrics, x_hat, y_hat in zip(branch_metrics, inputs_hat, outputs_hat):
        _, best_x, best_y = min(
            (initial[s0] + metrics[range(4), y].sum() + final[s1], x, y)
            for s0, x, y, s1 in all_paths(trellis)
        )
        np.testing.assert_equal(x_hat, best_x)
        np.testing.assert_equal(y_hat, best_y)


@pytest.mark.repeat(10)
def test_trellis_viterbi_shapes(rng):
    # Leading dimensions, with broadcast end metrics
    trellis = random_trellis(rng, 4)
    initial = rng.random((3, trellis[0].num_states))
    final = rng.random((3, trellis[-1].num_next_states))
    branch_metrics = rng.random((2, 3, 4, 12))
    inputs_hat = trellis.viterbi(branch_metrics, initial, final)
    assert inputs_hat.shape == (2, 3, 4)
    for i, j in product(range(2), range(3)):
        x_hat = trellis.viterbi(branch_metrics[i, j], initial[j], final[j])
        np.testing.assert_equal(inputs_hat[i, j], x_hat)


@pytest.mark.repeat(20)
def test_trellis_viterbi_ties(rng):
    # Integer costs, hence many ties
    trellis = random_trellis(rng, 4)
    initial = rng.integers(3, size=trellis[0].num_states)
    final = rng.integers(3, size=trellis[-1].num_next_states)
    branch_metrics = rng.integers(3, size=(4, 12))

    def key(path):
        # Least cost, then least final state, last branch, and so on
        s0, x, y, s1 = path
        cost = initial[s0] + branch_metrics[range(4), y].sum() + final[s1]
        return cost, s1, branches(trellis, s0, x)[::-1]

    _, best_x, best_y, _ = min(all_paths(trellis), key=key)
    args = (branch_metrics, initial, final)
    np.testing.assert_equal(trellis.viterbi(*args, symbols="input"), best_x)
    np.testing.assert_equal(trellis.viterbi(*args, symbols="output"), best_y)


@pytest.mark.repeat(10)
def test_trellis_forward_backward_brute_force(rng):
    trellis = random_trellis(rng, 4)
    # Repeated outputs, to be summed
    trellis = komm.Trellis([
        komm.TrellisSection(s.transitions, rng.integers(12, size=s.outputs.shape))
        for s in trellis
    ])
    initial = rng.normal(size=trellis[0].num_states)
    final = rng.normal(size=trellis[-1].num_next_states)
    branch_metrics = rng.normal(size=(5, 4, 12))
    args = (branch_metrics, initial, final)
    log_app_in = trellis.forward_backward(*args, symbols="input")
    log_app_out = trellis.forward_backward(*args, symbols="output")
    for metrics, app_in, app_out in zip(branch_metrics, log_app_in, log_app_out):
        expected_in, expected_out = np.full((4, 3), -np.inf), np.full((4, 12), -np.inf)
        for s0, inputs, outputs, s1 in all_paths(trellis):
            weight = initial[s0] + metrics[range(4), outputs].sum() + final[s1]
            steps = (range(4), inputs)
            expected_in[steps] = np.logaddexp(expected_in[steps], weight)
            steps = (range(4), outputs)
            expected_out[steps] = np.logaddexp(expected_out[steps], weight)
        normalizer = np.logaddexp.reduce(expected_in, axis=1, keepdims=True)
        np.testing.assert_allclose(app_in, expected_in - normalizer)
        np.testing.assert_allclose(app_out, expected_out - normalizer)


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "symbols, width",
    [("input", 3), ("output", 12)],
)
def test_trellis_forward_backward_shapes(rng, symbols, width):
    # Leading dimensions, with broadcast end metrics
    trellis = random_trellis(rng, 4)
    initial = rng.normal(size=(3, trellis[0].num_states))
    final = rng.normal(size=(3, trellis[-1].num_next_states))
    branch_metrics = rng.normal(size=(2, 3, 4, 12))
    log_app = trellis.forward_backward(branch_metrics, initial, final, symbols)
    assert log_app.shape == (2, 3, 4, width)
    for i, j in product(range(2), range(3)):
        args = (branch_metrics[i, j], initial[j], final[j], symbols)
        np.testing.assert_allclose(log_app[i, j], trellis.forward_backward(*args))


def test_trellis_single_parity_check(rng):
    # Code (3, 2); state is partial parity
    trellis = komm.Trellis([
        komm.TrellisSection([[0, 1]]),
        komm.TrellisSection([[0, 1], [1, 0]]),
        komm.TrellisSection([[0, -1], [-1, 0]]),
    ])
    r = rng.normal(size=(100, 3))
    zeros = np.zeros_like(r)
    v_hat = trellis.viterbi(np.stack([zeros, r], axis=2))
    decoder = komm.WagnerDecoder(komm.SingleParityCheckCode(3))
    np.testing.assert_equal(v_hat, decoder.decode_to_codeword(r))
    branch_metrics = np.stack([zeros, -r], axis=2)
    log_app_in = trellis.forward_backward(branch_metrics, symbols="input")
    extrinsic = boxplus(r[:, [1, 2, 0]], r[:, [2, 0, 1]])
    np.testing.assert_allclose(log_app_in[..., 0] - log_app_in[..., 1], r + extrinsic)
    log_app_out = trellis.forward_backward(branch_metrics, symbols="output")
    np.testing.assert_allclose(log_app_out, log_app_in)  # Single label


def test_trellis_viterbi_sklar():
    # [Skl01, pp. 401–405]
    section = komm.TrellisSection(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
    )
    received = np.array([[1, 1], [0, 1], [0, 1], [1, 0], [0, 1]])
    distances = np.count_nonzero(received[:, np.newaxis] != BITS, axis=2)
    initial = [0.0, np.inf, np.inf, np.inf]
    inputs, costs = survivors(komm.Trellis([section] * 5), distances, initial)
    np.testing.assert_equal(costs, [2, 2, 2, 1])
    np.testing.assert_equal(
        inputs,
        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 1, 1, 0], [1, 1, 0, 1, 1]],
    )


def test_trellis_viterbi_ryan_lin():
    # [RL09, pp. 176–177]
    section = komm.TrellisSection(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        outputs=[[0, 3], [1, 2], [3, 0], [2, 1]],
    )
    received = np.array([[-0.7, -0.5], [-0.8, -0.6], [-1.1, +0.4], [+0.9, +0.8]])
    correlations = received @ (-1) ** BITS.T
    initial = [0.0, np.inf, np.inf, np.inf]
    inputs, costs = survivors(komm.Trellis([section] * 4), -correlations, initial)
    np.testing.assert_allclose(costs, [-3.8, -3.4, -2.6, -2.4])
    np.testing.assert_equal(
        inputs, [[1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
    )


def test_trellis_forward_backward_lin_costello():
    # [LC04, pp. 572–575]
    section = komm.TrellisSection([[0, 1], [1, 0]], [[0, 3], [2, 1]])
    received = np.array([[-0.8, -0.1], [-1.0, +0.5], [+1.8, -1.1], [-1.6, +1.6]])
    branch_metrics = 0.5 * received @ (-1) ** BITS.T
    ends = [0.0, -np.inf]
    log_app = komm.Trellis([section] * 4).forward_backward(branch_metrics, ends, ends)
    np.testing.assert_allclose(
        log_app[:, 1] - log_app[:, 0], [0.48, 0.62, -1.02, 2.08], atol=0.05
    )


def test_trellis_forward_backward_abrantes():
    # [Abr10, pp. 434–437]
    section = komm.TrellisSection(
        transitions=[[0, 2], [0, 2], [1, 3], [1, 3]],
        outputs=[[0, 3], [3, 0], [1, 2], [2, 1]],
    )
    received = np.array([
        [-0.3, -0.1],
        [+0.5, -0.2],
        [-0.8, -0.5],
        [+0.5, -0.3],
        [-0.1, +0.7],
        [-1.5, +0.4],
    ])
    branch_metrics = 2.5 * received @ (-1) ** BITS.T
    ends = [0.0, -np.inf, -np.inf, -np.inf]
    log_app = komm.Trellis([section] * 6).forward_backward(branch_metrics, ends, ends)
    expected = [1.78, 0.24, -1.97, 5.52, -np.inf, -np.inf]
    np.testing.assert_allclose(log_app[:, 1] - log_app[:, 0], expected, atol=0.05)


@pytest.mark.parametrize("method", ["viterbi", "forward_backward"])
def test_trellis_invalid_symbols(method):
    trellis = komm.Trellis([SECTION_0])
    with pytest.raises(ValueError, match="'symbols' must be 'input' or 'output'"):
        getattr(trellis, method)(np.zeros((1, 4)), symbols="inputs")
