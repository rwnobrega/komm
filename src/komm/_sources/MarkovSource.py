from bisect import bisect_right
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._util import global_rng
from .._util.information_theory import entropy
from .._util.validators import (
    validate_float_array,
    validate_integer_array,
    validate_nonnegative_integer,
    validate_pmf,
    validate_transition_matrix,
)
from ..types import Array1D, Array2D
from .MarkovChain import MarkovChain


class MarkovSource:
    r"""
    Markov source. It is defined by a finite *set of states* $\mathcal{S}$, a finite *source alphabet* $\mathcal{X}$, a *transition function* $T : \mathcal{S} \times \mathcal{X} \to \mathcal{S}$, a *conditional pmf* $p(x \mid s)$ of the symbol $x \in \mathcal{X}$ given the state $s \in \mathcal{S}$, and an *initial pmf* $\pi_0$ over $\mathcal{S}$. The initial state $S_0$ has pmf $\pi_0$. At each time $t$, the source emits the symbol $X_t$ with pmf $p(\cdot \mid S_t)$ and goes to the state $S_{t+1} = T(S_t, X_t)$. So the states form a Markov chain, and the initial state and the emitted symbols determine all the states. Here, for simplicity, the set of states and the source alphabet are taken as $\mathcal{S} = [0 : |\mathcal{S}|)$ and $\mathcal{X} = [0 : |\mathcal{X}|)$, respectively, where $|\mathcal{X}|$ is called the *source cardinality*. For more details, see <cite>Gal08, Sec. 2.8</cite>.

    Parameters:
        transitions: The matrix of transitions of the source, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $T(s, x) \in \mathcal{S}$, that is, the next state of the source given that the current state is $s$ and the emitted symbol is $x$.

        pmfs: The matrix of conditional pmfs of the source, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s \in \mathcal{S}$ and column $x \in \mathcal{X}$ should be $p(x \mid s)$, that is, the probability of emitting the symbol $x$ given that the current state is $s$. Its elements must be non-negative and each row must sum to $1$.

        initial_pmf: The initial pmf $\pi_0$ over $\mathcal{S}$. Must be a 1D-array of size $|\mathcal{S}|$. The default value is the stationary distribution $\pi$ of the chain of states, which must then be irreducible.

    Examples:
        1. Consider the Markov source whose state diagram is depicted in the figure below.

            <figure markdown>
            ![Markov source example.](/fig/markov_source.svg)
            </figure>

            It has the set of states $\mathcal{S} = \\{ 0, 1 \\}$ and source alphabet $\mathcal{X} = \\{ 0, 1 \\}$. The transition function $T$ and the conditional pmf $p$ are given by the table below. Since $p(0 \mid 1) = 0$, the value of $T(1, 0)$ is arbitrary.

            | State $s$ | Symbol $x$ | Transition $T(s, x)$ | Probability $p(x \mid s)$ |
            | :-------: | :--------: | :------------------: | :-----------------------: |
            | $0$       | $0$        | $0$                  | $0.5$                     |
            | $0$       | $1$        | $1$                  | $0.5$                     |
            | $1$       | $0$        | $0$                  | $0$                       |
            | $1$       | $1$        | $0$                  | $1$                       |

            In other words, the ones come in pairs, so that every run of ones between two zeros has even length. The output is not a Markov chain of any order, since the next symbol depends on the parity of the current run of ones, which may be arbitrarily long.

                >>> source = komm.MarkovSource(
                ...     transitions=[[0, 1], [0, 0]],
                ...     pmfs=[[0.5, 0.5], [0, 1]],
                ... )
    """

    def __init__(
        self,
        transitions: npt.ArrayLike,
        pmfs: npt.ArrayLike,
        initial_pmf: npt.ArrayLike | None = None,
        rng: np.random.Generator | None = None,
    ):
        transitions = validate_integer_array(transitions, ndim=2)
        self._transitions = validate_integer_array(
            transitions, low=0, high=transitions.shape[0]
        )
        pmfs = validate_float_array(pmfs, shape=transitions.shape)
        self._pmfs = validate_transition_matrix(pmfs)
        if initial_pmf is not None:
            initial_pmf = validate_pmf(initial_pmf, size=self.num_states)
        elif not self.chain.is_irreducible():
            raise ValueError("'initial_pmf' is required for reducible chains")
        self._initial_pmf = initial_pmf
        self._rng = rng

    @property
    def rng(self) -> np.random.Generator:
        return global_rng.get() if self._rng is None else self._rng

    def __repr__(self) -> str:
        args = [
            f"transitions={self.transitions.tolist()}",
            f"pmfs={self.pmfs.tolist()}",
        ]
        if self._initial_pmf is not None:
            args.append(f"initial_pmf={self._initial_pmf.tolist()}")
        return f"{self.__class__.__name__}({', '.join(args)})"

    @property
    def transitions(self) -> Array2D[np.integer]:
        return self._transitions

    @property
    def pmfs(self) -> Array2D[np.floating]:
        return self._pmfs

    @property
    def initial_pmf(self) -> Array1D[np.floating]:
        r"""
        The initial pmf $\pi_0$ over $\mathcal{S}$.

        Examples:
            >>> source = komm.MarkovSource(
            ...     transitions=[[0, 1], [0, 0]],
            ...     pmfs=[[0.5, 0.5], [0, 1]],
            ... )
            >>> source.initial_pmf
            array([0.66666667, 0.33333333])

            >>> source = komm.MarkovSource(
            ...     transitions=[[0, 1], [0, 0]],
            ...     pmfs=[[0.5, 0.5], [0, 1]],
            ...     initial_pmf=[1, 0],
            ... )
            >>> source.initial_pmf
            array([1., 0.])
        """
        if self._initial_pmf is None:
            return self._stationary_pmf
        return self._initial_pmf

    @property
    def num_states(self) -> int:
        r"""
        The number of states of the source, $|\mathcal{S}|$.

        Examples:
            >>> source = komm.MarkovSource(
            ...     transitions=[[0, 1], [0, 0]],
            ...     pmfs=[[0.5, 0.5], [0, 1]],
            ... )
            >>> source.num_states
            2
        """
        return self._transitions.shape[0]

    @property
    def cardinality(self) -> int:
        r"""
        The source cardinality $|\mathcal{X}|$.

        Examples:
            >>> source = komm.MarkovSource(
            ...     transitions=[[0, 1], [0, 0]],
            ...     pmfs=[[0.5, 0.5], [0, 1]],
            ... )
            >>> source.cardinality
            2
        """
        return self._transitions.shape[1]

    @cached_property
    def chain(self) -> MarkovChain:
        r"""
        The [Markov chain](/ref/MarkovChain) of the states. Its transition matrix is given by
        $$
            P_{s,s'} = \sum_{x \in \mathcal{X} : \ T(s, x) = s'} p(x \mid s).
        $$

        Examples:
            >>> source = komm.MarkovSource(
            ...     transitions=[[0, 1], [0, 0]],
            ...     pmfs=[[0.5, 0.5], [0, 1]],
            ... )
            >>> source.chain.transition_matrix
            array([[0.5, 0.5],
                   [1. , 0. ]])
        """
        n = self.num_states
        matrix = np.zeros((n, n))
        np.add.at(matrix, (np.arange(n)[:, np.newaxis], self.transitions), self.pmfs)
        return MarkovChain(matrix)

    @cached_property
    def _stationary_pmf(self) -> Array1D[np.floating]:
        return self.chain.stationary_distribution()

    def entropy_rate(self, base: float | Literal["e"] = 2.0) -> float:
        r"""
        Computes the source entropy rate. For a Markov source, it is given by
        $$
            \mathrm{H} = \sum_{s \in \mathcal{S}} \pi(s) \, \mathrm{H}(p(\cdot \mid s)),
        $$
        where $\pi$ is the stationary distribution of the chain of states, and $\mathrm{H}(p(\cdot \mid s))$ is the [entropy](/ref/entropy) of the conditional pmf given the state $s$. It does not depend on the initial pmf. See <cite>Gal08, Sec. 2.8.2</cite>.

        Note:
            This method is only implemented for irreducible chains.

        Parameters:
            base: See [`komm.entropy`](/ref/entropy). The default value is `2.0`.

        Returns:
            entropy_rate: The source entropy rate.

        Examples:
            >>> source = komm.MarkovSource(
            ...     transitions=[[0, 1], [0, 0]],
            ...     pmfs=[[0.5, 0.5], [0, 1]],
            ... )
            >>> source.entropy_rate()  # doctest: +FLOAT_CMP
            0.6666666666666666
        """
        q = self._stationary_pmf
        return float(q @ [entropy(pmf, base) for pmf in self.pmfs])

    @cached_property
    def _cdfs(self) -> list[list[float]]:
        # Same normalization as `Generator.choice`
        cdfs = np.cumsum(self.pmfs, axis=1)
        cdfs /= cdfs[:, -1:]
        return cdfs.tolist()

    @cached_property
    def _initial_cdf(self) -> list[float]:
        cdf = np.cumsum(self.initial_pmf)
        cdf /= cdf[-1]
        return cdf.tolist()

    def emit(self, steps: int) -> Array1D[np.integer]:
        r"""
        Returns random symbols from the source. Each call starts from a new initial state, drawn from the initial pmf $\pi_0$.

        Parameters:
            steps: The number of symbols to emit. Must be a non-negative integer.

        Returns:
            symbols: A 1D-array with the emitted symbols, with values in $\mathcal{X}$.

        Examples:
            >>> source = komm.MarkovSource(
            ...     transitions=[[0, 1], [0, 0]],
            ...     pmfs=[[0.5, 0.5], [0, 1]],
            ... )
            >>> source.emit(20)
            array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
        """
        steps = validate_nonnegative_integer(steps)
        cdfs, transitions = self._cdfs, self.transitions.tolist()
        state = bisect_right(self._initial_cdf, self.rng.random())
        symbols: list[int] = []
        for u in self.rng.random(steps).tolist():
            symbol = bisect_right(cdfs[state], u)
            symbols.append(symbol)
            state = transitions[state][symbol]
        return np.array(symbols, dtype=int)
