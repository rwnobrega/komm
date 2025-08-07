from itertools import product

import numpy as np
import pytest

import komm


def test_markov_chain_grinstead_snell_oz():
    chain = komm.MarkovChain([
        [0.50, 0.25, 0.25],
        [0.50, 0.00, 0.50],
        [0.25, 0.25, 0.50],
    ])
    assert chain.communicating_classes() == [{0, 1, 2}]
    assert chain.is_irreducible() is True
    assert chain.transient_states() == set()
    assert chain.recurrent_states() == {0, 1, 2}
    assert chain.is_regular() is True
    assert chain.index_of_primitivity() == 2
    np.testing.assert_allclose(chain.stationary_distribution(), [0.4, 0.2, 0.4])


def test_markov_chain_grinstead_snell_drunk():
    chain = komm.MarkovChain([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    assert chain.communicating_classes() == [{0}, {1, 2, 3}, {4}]
    assert chain.is_irreducible() is False
    assert chain.transient_states() == {1, 2, 3}
    assert chain.absorbing_states() == {0, 4}
    np.testing.assert_allclose(
        chain.mean_number_of_visits(),
        [[1.5, 1.0, 0.5], [1.0, 2.0, 1.0], [0.5, 1.0, 1.5]],
    )
    np.testing.assert_allclose(
        chain.mean_time_to_absorption(),
        [3, 4, 3],
    )
    np.testing.assert_allclose(
        chain.absorption_probabilities(),
        [[0.75, 0.25], [0.50, 0.50], [0.25, 0.75]],
    )


def test_markov_chain_yates_ex_classes():
    # [YG.14, Examples 14, 15]
    chain = komm.MarkovChain([
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ])
    assert chain.communicating_classes() == [{0, 1, 2}, {3}, {4, 5}]
    assert chain.is_irreducible() is False
    assert chain.transient_states() == {3}
    assert chain.recurrent_states() == {0, 1, 2, 4, 5}


def test_markov_chain_yates_ex_packet_voice():
    # [YG.14, Examples 2, 8, 16]
    chain = komm.MarkovChain([[139 / 140, 1 / 140], [1 / 100, 99 / 100]])
    np.testing.assert_allclose(chain.stationary_distribution(), [7 / 12, 5 / 12])


def test_markov_chain_pishro_nik_1():
    # [https://www.probabilitycourse.com/chapter11/11_2_4_classification_of_states.php]
    # Examples 11.6 and 11.8.
    chain = komm.MarkovChain([
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])
    assert chain.communicating_classes() == [{0, 1}, {2, 3}, {4}, {5, 6, 7}]
    assert chain.is_irreducible() is False
    assert chain.transient_states() == {0, 1, 2, 3, 4}
    assert chain.recurrent_states() == {5, 6, 7}
    assert chain.period(0) == 1
    assert chain.period(2) == 2
    assert chain.period(5) == 1


def test_markov_chain_pishro_nik_2():
    # [https://www.probabilitycourse.com/chapter11/11_2_5_using_the_law_of_total_probability_with_recursion.php]
    chain = komm.MarkovChain([
        [1, 0, 0, 0],
        [1 / 3, 0, 2 / 3, 0],
        [0, 1 / 2, 0, 1 / 2],
        [0, 0, 0, 1],
    ])
    assert chain.communicating_classes() == [{0}, {1, 2}, {3}]
    assert chain.is_irreducible() is False
    assert chain.transient_states() == {1, 2}
    assert chain.recurrent_states() == {0, 3}
    assert chain.absorbing_states() == {0, 3}
    np.testing.assert_allclose(
        chain.mean_time_to_absorption(),
        [5 / 2, 9 / 4],
    )
    np.testing.assert_allclose(
        chain.absorption_probabilities(),
        [[1 / 2, 1 / 2], [1 / 4, 3 / 4]],
    )


@pytest.mark.parametrize(
    "transition_matrix",
    [
        [[1, 0], [0, 1]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    ],
)
def test_markov_chain_index_of_primitivity_not_regular(transition_matrix):
    chain = komm.MarkovChain(transition_matrix)
    with pytest.raises(ValueError, match="chain is not regular"):
        chain.index_of_primitivity()


@pytest.mark.parametrize("n", range(2, 10))
def test_markov_chain_index_of_primitivity_tight(n):
    # Wielandt's construction
    # See [Meyer - Matrix Analysis and Applied Linear Algebra, Example 8.3.9, p. 685]
    P = np.eye(n, k=1)
    P[-1, 0] = P[-1, 1] = 0.5
    chain = komm.MarkovChain(P)
    assert chain.index_of_primitivity() == n**2 - 2 * n + 2
