import numpy as np

from scipy.special import logsumexp


__all__ = ['FiniteStateMachine']


class FiniteStateMachine:
    """
    Finite-state machine (Mealy machine). It is defined by a *set of states* :math:`\\mathcal{S}`, an *input alphabet* :math:`\\mathcal{X}`, an *output alphabet* :math:`\\mathcal{Y}`, and a *transition function* :math:`T : \\mathcal{S} \\times \\mathcal{X} \\to \\mathcal{S} \\times \\mathcal{Y}`. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are always taken as :math:`\\mathcal{S} = \\{ 0, 1, \\ldots, |\\mathcal{S}| - 1 \\}`, :math:`\\mathcal{X} = \\{ 0, 1, \\ldots, |\\mathcal{X}| - 1 \\}`, and :math:`\\mathcal{Y} = \\{ 0, 1, \\ldots, |\\mathcal{Y}| - 1 \\}`, respectively.

    For example, consider the finite-state machine whose state diagram depicted in the figure below.

    .. image:: figures/mealy.png
       :alt: Finite-state machine (Mealy machine) example.
       :align: center

    It has set of states :math:`\\mathcal{S} = \\{ 0, 1, 2, 3 \\}`, input alphabet :math:`\\mathcal{X} = \\{ 0, 1 \\}`, output alphabet :math:`\\mathcal{Y} = \\{ 0, 1, 2, 3 \\}`, and transition function :math:`T` given by the table below.

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
        """
        Constructor for the class. It expects the following parameters:

        :code:`next_states` : 2D-array of :obj:`int`
            The matrix of next states of the machine, of shape :math:`|\\mathcal{S}| \\times |\\mathcal{X}|`. The element in row :math:`s` and column :math:`x` should be the next state of the machine (an element in :math:`\\mathcal{S}`), given that the current state is :math:`s \\in \\mathcal{S}` and the input is :math:`x \\in \\mathcal{X}`.

        :code:`outputs` : 2D-array of :obj:`int`
            The matrix of outputs of the machine, of shape :math:`|\\mathcal{S}| \\times |\\mathcal{X}|`. The element in row :math:`s` and column :math:`x` should be the output of the machine (an element in :math:`\\mathcal{Y}`), given that the current state is :math:`s \\in \\mathcal{S}` and the input is :math:`x \\in \\mathcal{X}`.

        .. rubric:: Examples

        >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
        """
        self._next_states = np.array(next_states, dtype=np.int)
        self._outputs = np.array(outputs, dtype=np.int)
        self._num_states, self._num_input_symbols = self._next_states.shape
        self._num_output_symbols = np.amax(self._outputs)

        self._input_edges = np.full((self._num_states, self._num_states), fill_value=-1)
        self._output_edges = np.full((self._num_states, self._num_states), fill_value=-1)
        for state_from in range(self._num_states):
            for x, state_to in enumerate(self._next_states[state_from, :]):
                self._input_edges[state_from, state_to] = x
                self._output_edges[state_from, state_to] = self._outputs[state_from, x]

    def __repr__(self):
        args = 'next_states={}, outputs={}'.format(self._next_states.tolist(), self._outputs.tolist())
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def num_states(self):
        """
        The number of states of the machine. This property is read-only.
        """
        return self._num_states

    @property
    def num_input_symbols(self):
        """
        The size (cardinality) of the input alphabet :math:`\\mathcal{X}`. This property is read-only.
        """
        return self._num_input_symbols

    @property
    def num_output_symbols(self):
        """
        The size (cardinality) of the output alphabet :math:`\\mathcal{Y}`. This property is read-only.
        """
        return self._num_output_symbols

    @property
    def next_states(self):
        """
        The matrix of next states of the machine. It has shape :math:`|\\mathcal{S}| \\times |\\mathcal{X}|`. The element in row :math:`s` and column :math:`x` is the next state of the machine (an element in :math:`\\mathcal{S}`), given that the current state is :math:`s \\in \\mathcal{S}` and the input is :math:`x \\in \\mathcal{X}`. This property is read-only.
        """
        return self._next_states

    @property
    def outputs(self):
        """
        The matrix of outputs of the machine. It has shape :math:`|\\mathcal{S}| \\times |\\mathcal{X}|`. The element in row :math:`s` and column :math:`x` is the output of the machine (an element in :math:`\\mathcal{Y}`), given that the current state is :math:`s \\in \\mathcal{S}` and the input is :math:`x \\in \\mathcal{X}`. This property is read-only.
        """
        return self._outputs

    @property
    def input_edges(self):
        """
        The matrix of input edges of the machine. It has shape :math:`|\\mathcal{S}| \\times |\\mathcal{S}|`. If there is an edge from :math:`s_0 \\in \\mathcal{S}` to :math:`s_1 \\in \\mathcal{S}`, then the element in row :math:`s_0` and column :math:`s_1` is the input associated with that edge (an element of :math:`\\mathcal{X}`); if there is no such edge, then the element is :math:`-1`. This property is read-only.

        .. rubric:: Example

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
        """
        The matrix of output edges of the machine. It has shape :math:`|\\mathcal{S}| \\times |\\mathcal{S}|`. If there is an edge from :math:`s_0 \\in \\mathcal{S}` to :math:`s_1 \\in \\mathcal{S}`, then the element in row :math:`s_0` and column :math:`s_1` is the output associated with that edge (an element of :math:`\\mathcal{Y}`); if there is no such edge, then the element is :math:`-1`. This property is read-only.

        .. rubric:: Example

        >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
        >>> fsm.output_edges
        array([[ 0,  3, -1, -1],
               [-1, -1,  1,  2],
               [ 3,  0, -1, -1],
               [-1, -1,  2,  1]])
        """
        return self._output_edges

    def process(self, input_sequence, initial_state):
        """
        Returns the output sequence corresponding to a given input sequence. It assumes the machine starts at a given initial state :math:`s_\\mathrm{i}`. The input sequence and the output sequence are denoted by :math:`\\mathbf{x} = (x_0, x_1, \\ldots, x_{L-1}) \\in \\mathcal{X}^L` and :math:`\\mathbf{y} = (y_0, y_1, \\ldots, y_{L-1}) \\in \\mathcal{Y}^L`, respectively.

        **Input:**

        :code:`input_sequence` : 1D-array of :obj:`int`
            The input sequence :math:`\\mathbf{x} \\in \\mathcal{X}^L`. It should be a 1D-array with elements in :math:`\\mathcal{X}`.

        :code:`initial_state` : :obj:`int`
            The initial state :math:`s_\\mathrm{i}` of the machine. Should be an integer in :math:`\\mathcal{S}`.

        **Output:**

        :code:`output_sequence` : 1D-array of :obj:`int`
            The output sequence :math:`\\mathbf{y} \\in \\mathcal{Y}^L` corresponding to :code:`input_sequence`, assuming the machine starts at the state given by :code:`initial_state`. It is a 1D-array with elements in :math:`\\mathcal{Y}`.

        :code:`final_state` : :obj:`int`
            The final state :math:`s_\\mathrm{f}` of the machine. It is an integer in :math:`\\mathcal{S}`.

        .. rubric:: Example

        >>> fsm = komm.FiniteStateMachine(next_states=[[0,1], [2,3], [0,1], [2,3]], outputs=[[0,3], [1,2], [3,0], [2,1]])
        >>> input_sequence, initial_state = [1, 1, 0, 1, 0], 0
        >>> output_sequence, final_state = fsm.process(input_sequence, initial_state)
        >>> output_sequence
        array([3, 2, 2, 0, 1])
        >>> final_state
        2
        """
        output_sequence = np.empty_like(input_sequence, dtype=np.int)
        s = initial_state
        for t, x in np.ndenumerate(input_sequence):
            y = self._outputs[s, x]
            s = self._next_states[s, x]
            output_sequence[t] = y
        final_state = s
        return output_sequence, final_state

    def viterbi(self, observed_sequence, metric_function, initial_metrics=None):
        """
        Applies the Viterbi algorithm on a given observed sequence. The Viterbi algorithm finds the most probable input sequence :math:`\\hat{\\mathbf{x}}(s) \\in \\mathcal{X}^L` ending in state :math:`s`, for all :math:`s \\in \\mathcal{S}`, given an observed sequence :math:`\\mathbf{z} \\in \\mathcal{Z}^L`. It is assumed uniform input priors.

        References: :cite:`Lin.Costello.04` (Sec. 12.1).

        **Input:**

        :code:`observed_sequence` : 1D-array
            The observed sequence :math:`\\mathbf{z} \\in \\mathcal{Z}^L`.

        :code:`metric_function` : function
            The metric function :math:`\\mathcal{Y} \\times \\mathcal{Z} \\to \\mathbb{R}`.

        :code:`initial_metrics` : 1D-array of :obj:`float`, optional
            The initial metrics for each state. It must be a 1D-array of length :math:`|\\mathcal{S}|`. The default value is :code:`0.0` for all states.

        **Output:**

        :code:`input_sequences_hat` : 2D-array of :obj:`int`
            The most probable input sequence :math:`\\hat{\\mathbf{x}}(s) \\in \\mathcal{X}^L` ending in state :math:`s`, for all :math:`s \\in \\mathcal{S}`. It is a 2D-array of shape :math:`L \\times |\\mathcal{S}|`, in which column :math:`s` is equal to :math:`\\hat{\\mathbf{x}}(s)`.

        :code:`final_metrics` : 1D-array of :obj:`float`
            The final metrics for each state. It is a 1D-array of length :math:`|\\mathcal{S}|`.
        """
        L, num_states = len(observed_sequence), self._num_states
        choices = np.empty((L, num_states), dtype=np.int)
        metrics = np.full((L + 1, num_states), fill_value=np.inf)
        if initial_metrics is None:
            metrics[0, :] = np.zeros(num_states, dtype=np.float)
        else:
            metrics[0, :] = initial_metrics
        for t, z in enumerate(observed_sequence):
            for s0 in range(num_states):
                for (s1, y) in zip(self._next_states[s0], self._outputs[s0]):
                    candidate_metrics = metrics[t, s0] + metric_function(y, z)
                    if candidate_metrics < metrics[t + 1, s1]:
                        metrics[t + 1, s1] = candidate_metrics
                        choices[t, s1] = s0

        # Backtrack
        input_sequences_hat = np.empty((L, num_states), dtype=np.int)
        for final_state in range(num_states):
            s1 = final_state
            for t in reversed(range(L)):
                s0 = choices[t, s1]
                input_sequences_hat[t, final_state] = self._input_edges[s0, s1]
                s1 = s0

        return input_sequences_hat, metrics[L, :]

    def viterbi_streaming(self, observed_sequence, metric_function, memory):
        """
        Applies the streaming version of the Viterbi algorithm on a given observed sequence. The path memory (or traceback length) is denoted by :math:`\\tau`. It chooses the survivor with best metric and selects the information block on this path.

        References: :cite:`Lin.Costello.04` (Sec. 12.3).

        **Input:**

        :code:`observed_sequence` : 1D-array
            The observed sequence :math:`\\mathbf{z} \\in \\mathcal{Z}^L`.

        :code:`metric_function` : function
            The metric function :math:`\\mathcal{Y} \\times \\mathcal{Z} \\to \\mathbb{R}`.

        **Output:**

        :code:`input_sequence_hat` : 1D-array of :obj:`int`
            The most probable input sequence. It is a 1D-array of length :math:`L`.

        **Input and output:**

        :code:`memory` : :obj:`dict`
            The past metrics for each state. It must be a 2D-array of shape :math:`|\\mathcal{S}| \\times (\\tau + 1)`.
            The initial metrics for each state. It must be a 2D-array of shape :math:`|\\mathcal{S}| \\times (\\tau + 1)`.
        """
        num_states = self._num_states
        input_sequences_hat = np.empty(len(observed_sequence), dtype=np.int)
        for t, z in enumerate(observed_sequence):
            new_metrics = np.full(num_states, fill_value=np.inf)
            choices = np.zeros(num_states, dtype=np.int)
            for s0 in range(num_states):
                for (s1, y) in zip(self._next_states[s0], self._outputs[s0]):
                    candidate_metric = memory['metrics'][s0, -1] + metric_function(y, z)
                    if candidate_metric < new_metrics[s1]:
                        new_metrics[s1] = candidate_metric
                        choices[s1] = s0

            s_star = np.argmin(new_metrics)
            s0, s1 = memory['paths'][s_star, :2]
            input_sequences_hat[t] = self._input_edges[s0, s1]

            memory['metrics'] = np.roll(memory['metrics'], shift=-1, axis=1)
            memory['metrics'][:, -1] = new_metrics
            memory['paths'] = np.roll(memory['paths'], shift=-1, axis=1)

            paths_copy = np.copy(memory['paths'])
            for s1, s0 in enumerate(choices):
                memory['paths'][s1, :-1] = paths_copy[s0, :-1]
                memory['paths'][s1, -1] = s1

        return input_sequences_hat

    def forward_backward(self, observed_sequence, metric_function, input_priors=None, initial_state_distribution=None, final_state_distribution=None):
        """
        Applies the forward-backward algorithm on a given observed sequence. The forward-backward algorithm computes the posterior :term:`pmf` of each input :math:`x_0, x_1, \\ldots, x_{L-1} \\in \\mathcal{X}` given an observed sequence :math:`\\mathbf{z} = (z_0, z_1, \\ldots, z_{L-1}) \\in \\mathcal{Z}^L`. The prior :term:`pmf` of each input may also be provided.

        References: :cite:`Lin.Costello.04` (Sec. 12.6).

        **Input:**

        :code:`observed_sequence` : 1D-array
            The observed sequence :math:`\\mathbf{z} \\in \\mathcal{Z}^L`.

        :code:`metric_function` : function
            The metric function :math:`\\mathcal{Y} \\times \\mathcal{Z} \\to \\mathbb{R}`.

        :code:`input_priors` : 2D-array of :obj:`float`, optional
            The prior :term:`pmf` of each input, of shape :math:`L \\times |\\mathcal{X}|`. The element in row :math:`t \\in [0 : L)` and column :math:`x \\in \\mathcal{X}` should be :math:`p(x_t = x)`. The default value considers uniform priors.

        :code:`initial_state_distribution` : 1D-array of :obj:`float`, optional
            The :term:`pmf` of the initial state of the machine. It must be a 1D-array of length :math:`|\\mathcal{S}|`. The default value is uniform over all states.

        :code:`final_state_distribution` : 1D-array of :obj:`float`, optional
            The :term:`pmf` of the final state of the machine. It must be a 1D-array of length :math:`|\\mathcal{S}|`. The default value is uniform over all states.

        **Output:**

        :code:`input_posteriors` : 2D-array of :obj:`float`
            The posterior :term:`pmf` of each input, given the observed sequence, of shape :math:`L \\times |\\mathcal{X}|`. The element in row :math:`t \\in [0 : L)` and column :math:`x \\in \\mathcal{X}` is :math:`p(x_t = x \\mid \\mathbf{z})`.
        """
        L, num_states, num_input_symbols = len(observed_sequence), self._num_states, self._num_input_symbols

        if input_priors is None:
            input_priors = np.ones((L, num_input_symbols)) / num_input_symbols
        if initial_state_distribution is None:
            initial_state_distribution = np.ones(num_states) / num_states
        if initial_state_distribution is None:
            final_state_distribution = np.ones(num_states) / num_states

        log_gamma = np.full((L, num_states, num_states), fill_value=-np.inf)
        log_alpha = np.full((L + 1, num_states), fill_value=-np.inf)
        log_beta = np.full((L + 1, num_states), fill_value=-np.inf)

        with np.errstate(divide='ignore'):
            log_input_priors = np.log(input_priors)
            log_alpha[0, :] = np.log(initial_state_distribution)
            log_beta[L, :] = np.log(final_state_distribution)

        for t, z in enumerate(observed_sequence):
            for x, s0 in np.ndindex(num_input_symbols, num_states):
                y, s1 = self._outputs[s0, x], self._next_states[s0, x]
                log_gamma[t, s0, s1] = log_input_priors[t, x] + metric_function(y, z)

        for t in range(0, L - 1):
            for s1 in range(num_states):
                log_alpha[t + 1, s1] = logsumexp(log_gamma[t, :, s1] + log_alpha[t, :])

        for t in range(L - 1, -1, -1):
            for s0 in range(num_states):
                log_beta[t, s0] = logsumexp(log_gamma[t, s0, :] + log_beta[t + 1, :])

        log_input_posteriors = np.empty((L, num_input_symbols), dtype=np.float)
        edge_labels = np.empty(num_states, dtype=np.float)
        for t in range(L):
            for x in range(num_input_symbols):
                for s0 in range(num_states):
                    s1 = self._next_states[s0, x]
                    edge_labels[s0] = log_alpha[t, s0] + log_gamma[t, s0, s1] + log_beta[t + 1, s1]
                log_input_posteriors[t, x] = logsumexp(edge_labels)

        input_posteriors = np.exp(log_input_posteriors - np.amax(log_input_posteriors, axis=1, keepdims=True))
        input_posteriors /= np.sum(input_posteriors, axis=1, keepdims=True)

        return input_posteriors
