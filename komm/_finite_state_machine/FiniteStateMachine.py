import numpy as np
import numpy.typing as npt
from scipy import special


class FiniteStateMachine:
    r"""
    Finite-state machine (Mealy machine). It is defined by a *set of states* $\mathcal{S}$, an *input alphabet* $\mathcal{X}$, an *output alphabet* $\mathcal{Y}$, and a *transition function* $T : \mathcal{S} \times \mathcal{X} \to \mathcal{S} \times \mathcal{Y}$. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are always taken as $\mathcal{S} = \\{ 0, 1, \ldots, |\mathcal{S}| - 1 \\}$, $\mathcal{X} = \\{ 0, 1, \ldots, |\mathcal{X}| - 1 \\}$, and $\mathcal{Y} = \\{ 0, 1, \ldots, |\mathcal{Y}| - 1 \\}$, respectively.

    For example, consider the finite-state machine whose state diagram depicted in the figure below.

    .. image:: figures/mealy.svg
       :alt: Finite-state machine (Mealy machine) example.
       :align: center

    It has set of states $\mathcal{S} = \\{ 0, 1, 2, 3 \\}$, input alphabet $\mathcal{X} = \\{ 0, 1 \\}$, output alphabet $\mathcal{Y} = \\{ 0, 1, 2, 3 \\}$, and transition function $T$ given by the table below.

    .. csv-table:: Transition function
       :align: center
       :header: State, Input, Next state, Output

       0, 0, 0, 0
       0, 1, 1, 3
       1, 0, 2, 1
       1, 1, 3, 2
       2, 0, 0, 3
       2, 1, 1, 0
       3, 0, 2, 2
       3, 1, 3, 1

    |
    """

    def __init__(self, next_states, outputs):
        r"""
        Constructor for the class.

        Parameters:

            next_states (2D-array of :obj:`int`): The matrix of next states of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s$ and column $x$ should be the next state of the machine (an element in $\mathcal{S}$), given that the current state is $s \in \mathcal{S}$ and the input is $x \in \mathcal{X}$.

            outputs (2D-array of :obj:`int`): The matrix of outputs of the machine, of shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s$ and column $x$ should be the output of the machine (an element in $\mathcal{Y}$), given that the current state is $s \in \mathcal{S}$ and the input is $x \in \mathcal{X}$.

        Examples:

            >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
        """
        self._next_states = np.array(next_states, dtype=int)
        self._outputs = np.array(outputs, dtype=int)
        self._num_states, self._num_input_symbols = self._next_states.shape
        self._num_output_symbols = np.amax(self._outputs)

        self._input_edges = np.full((self._num_states, self._num_states), fill_value=-1)
        self._output_edges = np.full((self._num_states, self._num_states), fill_value=-1)
        for state_from in range(self._num_states):
            for x, state_to in enumerate(self._next_states[state_from, :]):
                self._input_edges[state_from, state_to] = x
                self._output_edges[state_from, state_to] = self._outputs[state_from, x]

    def __repr__(self):
        args = "next_states={}, outputs={}".format(self._next_states.tolist(), self._outputs.tolist())
        return "{}({})".format(self.__class__.__name__, args)

    @property
    def num_states(self):
        r"""
        The number of states of the machine. This property is read-only.
        """
        return self._num_states

    @property
    def num_input_symbols(self):
        r"""
        The size (cardinality) of the input alphabet $\mathcal{X}$. This property is read-only.
        """
        return self._num_input_symbols

    @property
    def num_output_symbols(self):
        r"""
        The size (cardinality) of the output alphabet $\mathcal{Y}$. This property is read-only.
        """
        return self._num_output_symbols

    @property
    def next_states(self):
        r"""
        The matrix of next states of the machine. It has shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s$ and column $x$ is the next state of the machine (an element in $\mathcal{S}$), given that the current state is $s \in \mathcal{S}$ and the input is $x \in \mathcal{X}$. This property is read-only.
        """
        return self._next_states

    @property
    def outputs(self):
        r"""
        The matrix of outputs of the machine. It has shape $|\mathcal{S}| \times |\mathcal{X}|$. The element in row $s$ and column $x$ is the output of the machine (an element in $\mathcal{Y}$), given that the current state is $s \in \mathcal{S}$ and the input is $x \in \mathcal{X}$. This property is read-only.
        """
        return self._outputs

    @property
    def input_edges(self):
        r"""
        The matrix of input edges of the machine. It has shape $|\mathcal{S}| \times |\mathcal{S}|$. If there is an edge from $s_0 \in \mathcal{S}$ to $s_1 \in \mathcal{S}$, then the element in row $s_0$ and column $s_1$ is the input associated with that edge (an element of $\mathcal{X}$); if there is no such edge, then the element is $-1$. This property is read-only.

        Examples:

            >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
            >>> fsm.input_edges
            array([[ 0,  1, -1, -1],
                   [-1, -1,  0,  1],
                   [ 0,  1, -1, -1],
                   [-1, -1,  0,  1]])
        """
        return self._input_edges

    @property
    def output_edges(self):
        r"""
        The matrix of output edges of the machine. It has shape $|\mathcal{S}| \times |\mathcal{S}|$. If there is an edge from $s_0 \in \mathcal{S}$ to $s_1 \in \mathcal{S}$, then the element in row $s_0$ and column $s_1$ is the output associated with that edge (an element of $\mathcal{Y}$); if there is no such edge, then the element is $-1$. This property is read-only.

        Examples:

            >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
            >>> fsm.output_edges
            array([[ 0,  3, -1, -1],
                   [-1, -1,  1,  2],
                   [ 3,  0, -1, -1],
                   [-1, -1,  2,  1]])
        """
        return self._output_edges

    def process(self, input_sequence, initial_state):
        r"""
        Returns the output sequence corresponding to a given input sequence. It assumes the machine starts at a given initial state $s_\mathrm{i}$. The input sequence and the output sequence are denoted by $\mathbf{x} = (x_0, x_1, \ldots, x_{L-1}) \in \mathcal{X}^L$ and $\mathbf{y} = (y_0, y_1, \ldots, y_{L-1}) \in \mathcal{Y}^L$, respectively.

        Parameters:

            input_sequence (1D-array of :obj:`int`): The input sequence $\mathbf{x} \in \mathcal{X}^L$. It should be a 1D-array with elements in $\mathcal{X}$.

            initial_state (:obj:`int`): The initial state $s_\mathrm{i}$ of the machine. Should be an integer in $\mathcal{S}$.

        Returns:

            output_sequence (1D-array of :obj:`int`): The output sequence $\mathbf{y} \in \mathcal{Y}^L$ corresponding to `input_sequence`, assuming the machine starts at the state given by `initial_state`. It is a 1D-array with elements in $\mathcal{Y}$.

            final_state (:obj:`int`): The final state $s_\mathrm{f}$ of the machine. It is an integer in $\mathcal{S}$.

        Examples:

            >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
            >>> input_sequence, initial_state = [1, 1, 0, 1, 0], 0
            >>> output_sequence, final_state = fsm.process(input_sequence, initial_state)
            >>> output_sequence
            array([3, 2, 2, 0, 1])
            >>> final_state
            2
        """
        output_sequence = np.empty_like(input_sequence, dtype=int)
        s = initial_state
        for t, x in np.ndenumerate(input_sequence):
            y = self._outputs[s, x]
            s = self._next_states[s, x]
            output_sequence[t] = y
        final_state = s
        return output_sequence, final_state

    def viterbi(self, observed_sequence, metric_function, initial_metrics=None):
        r"""
        Applies the Viterbi algorithm on a given observed sequence. The Viterbi algorithm finds the most probable input sequence $\hat{\mathbf{x}}(s) \in \mathcal{X}^L$ ending in state $s$, for all $s \in \mathcal{S}$, given an observed sequence $\mathbf{z} \in \mathcal{Z}^L$. It is assumed uniform input priors. See :cite:`Lin.Costello.04` (Sec. 12.1).

        Parameters:

            observed_sequence (1D-array): The observed sequence $\mathbf{z} \in \mathcal{Z}^L$.

            metric_function (function): The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            initial_metrics (1D-array of :obj:`float`, optional): The initial metrics for each state. It must be a 1D-array of length $|\mathcal{S}|$. The default value is `0.0` for all states.

        Returns:

            input_sequences_hat (2D-array of :obj:`int`): The most probable input sequence $\hat{\mathbf{x}}(s) \in \mathcal{X}^L$ ending in state $s$, for all $s \in \mathcal{S}$. It is a 2D-array of shape $L \times |\mathcal{S}|$, in which column $s$ is equal to $\hat{\mathbf{x}}(s)$.

            final_metrics (1D-array of :obj:`float`): The final metrics for each state. It is a 1D-array of length $|\mathcal{S}|$.
        """
        L, num_states = len(observed_sequence), self._num_states
        choices = np.empty((L, num_states), dtype=int)
        metrics = np.full((L + 1, num_states), fill_value=np.inf)
        if initial_metrics is None:
            metrics[0, :] = np.zeros(num_states, dtype=float)
        else:
            metrics[0, :] = initial_metrics
        for t, z in enumerate(observed_sequence):
            for s0 in range(num_states):
                for s1, y in zip(self._next_states[s0], self._outputs[s0]):
                    candidate_metrics = metrics[t, s0] + metric_function(y, z)
                    if candidate_metrics < metrics[t + 1, s1]:
                        metrics[t + 1, s1] = candidate_metrics
                        choices[t, s1] = s0

        # Backtrack
        input_sequences_hat = np.empty((L, num_states), dtype=int)
        for final_state in range(num_states):
            s1 = final_state
            for t in reversed(range(L)):
                s0 = choices[t, s1]
                input_sequences_hat[t, final_state] = self._input_edges[s0, s1]
                s1 = s0

        return input_sequences_hat, metrics[L, :]

    def viterbi_streaming(self, observed_sequence, metric_function, memory):
        r"""
        Applies the streaming version of the Viterbi algorithm on a given observed sequence. The path memory (or traceback length) is denoted by $\tau$. It chooses the survivor with best metric and selects the information block on this path. See :cite:`Lin.Costello.04` (Sec. 12.3).

        Parameters:

            observed_sequence (1D-array): The observed sequence $\mathbf{z} \in \mathcal{Z}^L$.

            metric_function (function): The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            memory (:obj:`dict`): The metrics for each state. It must be a dictionary containing two keys: `'paths'`, a 2D-array of :obj:`int` of shape $|\mathcal{S}| \times (\tau + 1)$; and `'metrics'`, a 2D-array of :obj:`float` of shape $|\mathcal{S}| \times (\tau + 1)$. This dictionary is updated in-place by this method.

        Returns:

            input_sequence_hat (1D-array of :obj:`int`): The most probable input sequence $\hat{\mathbf{x}} \in \mathcal{X}^L$
        """
        num_states = self._num_states
        input_sequences_hat = np.empty(len(observed_sequence), dtype=int)
        for t, z in enumerate(observed_sequence):
            new_metrics = np.full(num_states, fill_value=np.inf)
            choices = np.zeros(num_states, dtype=int)
            for s0 in range(num_states):
                for s1, y in zip(self._next_states[s0], self._outputs[s0]):
                    candidate_metric = memory["metrics"][s0, -1] + metric_function(y, z)
                    if candidate_metric < new_metrics[s1]:
                        new_metrics[s1] = candidate_metric
                        choices[s1] = s0

            s_star = np.argmin(new_metrics)
            s0, s1 = memory["paths"][s_star, :2]
            input_sequences_hat[t] = self._input_edges[s0, s1]

            memory["metrics"] = np.roll(memory["metrics"], shift=-1, axis=1)
            memory["metrics"][:, -1] = new_metrics
            memory["paths"] = np.roll(memory["paths"], shift=-1, axis=1)

            paths_copy = np.copy(memory["paths"])
            for s1, s0 in enumerate(choices):
                memory["paths"][s1, :-1] = paths_copy[s0, :-1]
                memory["paths"][s1, -1] = s1

        return input_sequences_hat

    def forward_backward(
        self,
        observed_sequence,
        metric_function,
        input_priors=None,
        initial_state_distribution=None,
        final_state_distribution=None,
    ):
        r"""
        Applies the forward-backward algorithm on a given observed sequence. The forward-backward algorithm computes the posterior :term:`pmf` of each input $x_0, x_1, \ldots, x_{L-1} \in \mathcal{X}$ given an observed sequence $\mathbf{z} = (z_0, z_1, \ldots, z_{L-1}) \in \mathcal{Z}^L$. The prior :term:`pmf` of each input may also be provided. See :cite:`Lin.Costello.04` (Sec. 12.6).

        Parameters:

            observed_sequence (1D-array): The observed sequence $\mathbf{z} \in \mathcal{Z}^L$.

            metric_function (function): The metric function $\mathcal{Y} \times \mathcal{Z} \to \mathbb{R}$.

            input_priors (2D-array of :obj:`float`, optional): The prior :term:`pmf` of each input, of shape $L \times |\mathcal{X}|$. The element in row $t \in [0 : L)$ and column $x \in \mathcal{X}$ should be $p(x_t = x)$. The default value considers uniform priors.

            initial_state_distribution (1D-array of :obj:`float`, optional): The :term:`pmf` of the initial state of the machine. It must be a 1D-array of length $|\mathcal{S}|$. The default value is uniform over all states.

            final_state_distribution (1D-array of :obj:`float`, optional): The :term:`pmf` of the final state of the machine. It must be a 1D-array of length $|\mathcal{S}|$. The default value is uniform over all states.

        Returns:

            input_posteriors (2D-array of :obj:`float`): The posterior :term:`pmf` of each input, given the observed sequence, of shape $L \times |\mathcal{X}|$. The element in row $t \in [0 : L)$ and column $x \in \mathcal{X}$ is $p(x_t = x \mid \mathbf{z})$.
        """
        L, num_states, num_input_symbols = len(observed_sequence), self._num_states, self._num_input_symbols

        if input_priors is None:
            input_priors = np.ones((L, num_input_symbols)) / num_input_symbols
        if initial_state_distribution is None:
            initial_state_distribution = np.ones(num_states) / num_states
        if final_state_distribution is None:
            final_state_distribution = np.ones(num_states) / num_states

        log_gamma = np.full((L, num_states, num_states), fill_value=-np.inf)
        log_alpha = np.full((L + 1, num_states), fill_value=-np.inf)
        log_beta = np.full((L + 1, num_states), fill_value=-np.inf)

        with np.errstate(divide="ignore"):
            log_input_priors = np.log(input_priors)
            log_alpha[0, :] = np.log(initial_state_distribution)
            log_beta[L, :] = np.log(final_state_distribution)

        for t, z in enumerate(observed_sequence):
            for x, s0 in np.ndindex(num_input_symbols, num_states):
                y, s1 = self._outputs[s0, x], self._next_states[s0, x]
                log_gamma[t, s0, s1] = log_input_priors[t, x] + metric_function(y, z)

        for t in range(0, L - 1):
            for s1 in range(num_states):
                log_alpha[t + 1, s1] = special.logsumexp(log_gamma[t, :, s1] + log_alpha[t, :])

        for t in range(L - 1, -1, -1):
            for s0 in range(num_states):
                log_beta[t, s0] = special.logsumexp(log_gamma[t, s0, :] + log_beta[t + 1, :])

        log_input_posteriors = np.empty((L, num_input_symbols), dtype=float)
        edge_labels = np.empty(num_states, dtype=float)
        for t in range(L):
            for x in range(num_input_symbols):
                for s0 in range(num_states):
                    s1 = self._next_states[s0, x]
                    edge_labels[s0] = log_alpha[t, s0] + log_gamma[t, s0, s1] + log_beta[t + 1, s1]
                log_input_posteriors[t, x] = special.logsumexp(edge_labels)

        input_posteriors = np.exp(log_input_posteriors - np.amax(log_input_posteriors, axis=1, keepdims=True))
        input_posteriors /= np.sum(input_posteriors, axis=1, keepdims=True)

        return input_posteriors
