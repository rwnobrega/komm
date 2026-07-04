import numpy as np

import komm


def hamiltonian_plus_chord_chain():
    # Hamiltonian cycle 0 → 1 → 2 → 3 → 4 → 5 → 0 (length 6), plus the edge
    # 5 → 1, which creates the inner cycle 1 → 2 → 3 → 4 → 5 → 1 (length 5)
    # *not* passing through state 0. The return times of state 0 are
    # {6, 11, 12, 16, 17, ...}, whose gcd is 1: the chain is aperiodic.
    # Note that the smallest pair of return times witnessing gcd 1 is (6, 11),
    # and 11 > |S| = 6, so inspecting only n ≤ |S| is not enough.
    P = np.zeros((6, 6))
    P[0, 1] = 1.0
    P[1, 2] = 1.0
    P[2, 3] = 1.0
    P[3, 4] = 1.0
    P[4, 5] = 1.0
    P[5, 0] = 0.5
    P[5, 1] = 0.5
    return komm.MarkovChain(P)


def brute_force_period(P, state, horizon):
    # gcd of all return times n ≤ horizon with (P^n)[state, state] > 0.
    from functools import reduce
    from math import gcd

    A = (np.asarray(P) > 0).astype(int)
    Pn = np.eye(A.shape[0], dtype=int)
    times = []
    for n in range(1, horizon + 1):
        Pn = ((Pn @ A) > 0).astype(int)
        if Pn[state, state]:
            times.append(n)
    return reduce(gcd, times) if times else 0


def test_period_with_long_return_times():
    chain = hamiltonian_plus_chord_chain()
    P = chain.transition_matrix
    for state in range(6):
        assert chain.period(state) == brute_force_period(P, state, horizon=200)


def test_is_aperiodic_with_long_return_times():
    chain = hamiltonian_plus_chord_chain()
    assert chain.is_aperiodic() is True


def test_is_regular_consistent_with_index_of_primitivity():
    # The chain is irreducible and aperiodic, hence regular; indeed,
    # index_of_primitivity() succeeds (P^26 > 0). Therefore is_regular()
    # must not contradict it.
    chain = hamiltonian_plus_chord_chain()
    n = chain.index_of_primitivity()
    Pn = np.linalg.matrix_power(chain.transition_matrix, n)
    assert np.all(Pn > 0)
    assert chain.is_regular() is True


def test_period_still_correct_for_truly_periodic_chains():
    # Guard for the fix: gcd(6, 3) = 3 via a chord through state 0.
    # Cycle 0 → 1 → 2 → 3 → 4 → 5 → 0 plus edge 2 → 0 (cycle 0 → 1 → 2 → 0).
    P = np.zeros((6, 6))
    P[0, 1] = 1.0
    P[1, 2] = 1.0
    P[2, 3] = 0.5
    P[2, 0] = 0.5
    P[3, 4] = 1.0
    P[4, 5] = 1.0
    P[5, 0] = 1.0
    chain = komm.MarkovChain(P)
    assert chain.period(0) == 3
    assert chain.is_aperiodic() is False


def test_accessible_states_cache_not_corrupted_by_caller():
    chain = komm.MarkovChain([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected = set(chain.accessible_states_from(0))
    returned = chain.accessible_states_from(0)
    try:
        returned.add(99)  # caller misbehaves
    except AttributeError:
        pass  # returning an immutable object is also acceptable
    assert set(chain.accessible_states_from(0)) == expected


def test_communicating_classes_cache_not_corrupted_by_caller():
    chain = komm.MarkovChain([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected = [set(c) for c in chain.communicating_classes()]
    returned = chain.communicating_classes()
    try:
        returned.clear()  # caller misbehaves
    except AttributeError:
        pass
    assert [set(c) for c in chain.communicating_classes()] == expected
    # is_irreducible() depends on communicating_classes(); it must survive too
    assert chain.is_irreducible() is False


def test_transient_and_recurrent_states_cache_not_corrupted_by_caller():
    chain = komm.MarkovChain([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.0, 1.0],
    ])
    expected_t = set(chain.transient_states())
    expected_r = set(chain.recurrent_states())
    for method in (chain.transient_states, chain.recurrent_states):
        returned = method()
        try:
            returned.add(99)
        except AttributeError:
            pass
    assert set(chain.transient_states()) == expected_t
    assert set(chain.recurrent_states()) == expected_r
