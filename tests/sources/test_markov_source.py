from re import escape, fullmatch

import numpy as np
import pytest

import komm

T2 = [[0, 1], [1, 0]]
P2 = [[0.5, 0.5], [0.5, 0.5]]


def test_markov_source_gallager():
    # [Gal08, Sec. 2.8]
    source = komm.MarkovSource(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        pmfs=[[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]],
    )
    np.testing.assert_allclose(source.initial_pmf, [5 / 12, 1 / 12, 1 / 12, 5 / 12])
    entropy_rate = (10 * komm.binary_entropy(0.1) + 2) / 12
    np.testing.assert_allclose(source.entropy_rate(), entropy_rate)


def test_markov_source_frequencies():
    source = komm.MarkovSource(
        transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
        pmfs=[[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]],
    )
    x = source.emit(100_000)
    states = 2 * x[:-2] + x[1:-1]  # Last two symbols
    counts = np.zeros((4, 2))
    np.add.at(counts, (states, x[2:]), 1)
    state_freqs = counts.sum(axis=1) / counts.sum()
    np.testing.assert_allclose(state_freqs, source.initial_pmf, atol=0.02)
    symbol_freqs = counts / counts.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(symbol_freqs, source.pmfs, atol=0.02)


@pytest.mark.parametrize(
    "initial_pmf, pattern",
    [
        ([1, 0], "(0|11)*1?"),
        ([0, 1], "1(0|11)*1?"),
    ],
)
def test_markov_source_even_process(initial_pmf, pattern):
    # Runs of ones between zeros have even length
    source = komm.MarkovSource(
        transitions=[[0, 1], [0, 0]],
        pmfs=[[0.5, 0.5], [0, 1]],
        initial_pmf=initial_pmf,
    )
    assert fullmatch(pattern, "".join(map(str, source.emit(1000))))
    np.testing.assert_allclose(source.entropy_rate(), 2 / 3)


def test_markov_source_one_state():
    pmf = [1 / 2, 1 / 4, 1 / 8, 1 / 8]
    source = komm.MarkovSource(transitions=[[0, 0, 0, 0]], pmfs=[pmf])
    np.testing.assert_allclose(source.initial_pmf, [1.0])
    dms = komm.DiscreteMemorylessSource(pmf)
    np.testing.assert_allclose(source.entropy_rate(), dms.entropy_rate())


@pytest.mark.parametrize(
    "transitions, pmfs, initial_pmf, message",
    [
        ([0, 1], P2, None, "'transitions' must be a 2D-array"),
        ([[0, 2], [1, 0]], P2, None, "elements of 'transitions' must be in [0:2)"),
        (T2, [[0.5, 0.5]], None, "'pmfs' must have shape (2, 2) (got (1, 2))"),
        (T2, [[1.5, -0.5], [0.5, 0.5]], None, "'pmfs' must be non-negative"),
        (T2, [[0.5, 0.6], [0.5, 0.5]], None, "rows of 'pmfs' must sum to 1.0"),
        (T2, P2, [1, 0, 0], "'initial_pmf' must have size 2 (got 3)"),
        (T2, P2, [0.5, 0.6], "'initial_pmf' must sum to 1.0"),
        ([[0, 0], [1, 1]], P2, None, "'initial_pmf' is required for reducible chains"),
    ],
)
def test_markov_source_invalid(transitions, pmfs, initial_pmf, message):
    with pytest.raises(ValueError, match=escape(message)):
        komm.MarkovSource(transitions, pmfs, initial_pmf)


@pytest.mark.parametrize(
    "steps, error, message",
    [
        (-1, ValueError, "'steps' must be a non-negative integer"),
        (1.0, TypeError, "'steps' must be an integer"),
    ],
)
def test_markov_source_invalid_steps(steps, error, message):
    source = komm.MarkovSource(T2, P2)
    with pytest.raises(error, match=message):
        source.emit(steps)


def test_markov_source_emit_empty():
    symbols = komm.MarkovSource(T2, P2).emit(0)
    assert symbols.shape == (0,)
    assert np.issubdtype(symbols.dtype, np.integer)


def test_markov_source_entropy_rate_reducible():
    source = komm.MarkovSource([[0, 0], [1, 1]], P2, initial_pmf=[1, 0])
    assert source.emit(5).shape == (5,)
    with pytest.raises(NotImplementedError, match="irreducible chains"):
        source.entropy_rate()


def test_markov_source_repr():
    source = komm.MarkovSource([[0, 0]], [[0.5, 0.5]])
    assert repr(source) == "MarkovSource(transitions=[[0, 0]], pmfs=[[0.5, 0.5]])"
    source = komm.MarkovSource([[0, 0]], [[0.5, 0.5]], initial_pmf=[1])
    assert repr(source).endswith(", initial_pmf=[1.0])")


def test_markov_source_follows_global_rng():
    source = komm.MarkovSource(T2, P2)
    komm.global_rng.set(np.random.default_rng(1))
    x = source.emit(100)
    komm.global_rng.set(np.random.default_rng(1))
    np.testing.assert_equal(source.emit(100), x)
