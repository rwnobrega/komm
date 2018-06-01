import numpy as np

from scipy.special import logsumexp

from ._algebra import \
    BinaryPolynomial, BinaryPolynomialFraction

from ._error_control_block import \
    BlockCode

from .util import \
    int2binlist, binlist2int, pack, unpack

from ._aux import tag

__all__ = ['FiniteStateMachine', 'ConvolutionalCode', 'TerminatedConvolutionalCode']


class FiniteStateMachine:
    """
    Finite-state machine (Mealy machine). It is defined by a *set of states* :math:`\\mathcal{S}`, a *initial state* :math:`s_\\mathrm{i} \\in \\mathcal{S}`, an *input alphabet* :math:`\\mathcal{X}`, an *output alphabet* :math:`\\mathcal{Y}`, and a *transition function* :math:`T : \\mathcal{S} \\times \\mathcal{X} \\to \\mathcal{S} \\times \\mathcal{Y}`. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are always taken as :math:`\\mathcal{S} = \\{ 0, 1, \ldots, |\\mathcal{S}| - 1 \\}`, :math:`\\mathcal{X} = \\{ 0, 1, \ldots, |\\mathcal{X}| - 1 \\}`, and :math:`\\mathcal{Y} = \\{ 0, 1, \ldots, |\\mathcal{Y}| - 1 \\}`, respectively.

    For example, consider the finite-state machine whose state diagram depicted in the figure below.

    .. image:: figures/mealy.png
       :alt: Finite-state machine (Mealy machine) example.
       :align: center

    It has set of states :math:`\\mathcal{S} = \\{ 0, 1, 2, 3 \\}`, initial state :math:`s_\\mathrm{i} = 0`, input alphabet :math:`\\mathcal{X} = \\{ 0, 1 \\}`, output alphabet :math:`\\mathcal{Y} = \\{ 0, 1, 2, 3 \\}`, and transition function :math:`T` given by the table below.

    .. csv-table:: Transition function
       :align: center
       :header: State, Input, State, Output

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
    def __init__(self, next_states, outputs, initial_state=0):
        """
        Constructor for the class. It expects the following parameters:

        :code:`next_states` : 2D-array of :obj:`int`
            The matrix of next states of the machine, of shape :math:`|\\mathcal{S}| \\times |\\mathcal{X}|`. The element in row :math:`s` and column :math:`x` should be the next state of the machine (an element in :math:`\\mathcal{S}`), given that the current state is :math:`s \\in \\mathcal{S}` and the input is :math:`x \\in \\mathcal{X}`.

        :code:`outputs` : 2D-array of :obj:`int`
            The matrix of outputs of the machine, of shape :math:`|\\mathcal{S}| \\times |\\mathcal{X}|`. The element in row :math:`s` and column :math:`x` should be the output of the machine (an element in :math:`\\mathcal{Y}`), given that the current state is :math:`s \\in \\mathcal{S}` and the input is :math:`x \\in \\mathcal{X}`.

        :code:`initial_state` : :obj:`int`, optional
            The initial state :math:`s_\\mathrm{i}` of the machine. Should be an integer in :math:`\\mathcal{S}`. The default value is :code:`0`.

        .. rubric:: Examples

        >>> fsm = komm.FiniteStateMachine(next_states=[[0, 1], [2, 3], [0, 1], [2, 3]], outputs=[[0, 3], [1, 2], [3, 0], [2, 1]])
        >>> fsm.process([1, 1, 0, 1, 0])
        array([3, 2, 2, 0, 1])
        >>> fsm.process([1, 1, 0, 1, 0])
        array([0, 2, 2, 0, 1])
        """
        self._next_states = np.array(next_states, dtype=np.int)
        self._outputs = np.array(outputs, dtype=np.int)
        self._num_states, self._num_input_symbols = self._next_states.shape
        self._num_output_symbols = np.amax(self._outputs)
        self._state = initial_state

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
    def state(self):
        """
        The current state of the machine. This is a read-and-write property.
        """
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

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

        >>> fsm = komm.FiniteStateMachine(next_states=[[0, 1], [2, 3], [0, 1], [2, 3]], outputs=[[0, 3], [1, 2], [3, 0], [2, 1]])
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

        >>> fsm = komm.FiniteStateMachine(next_states=[[0, 1], [2, 3], [0, 1], [2, 3]], outputs=[[0, 3], [1, 2], [3, 0], [2, 1]])
        >>> fsm.output_edges
        array([[ 0,  3, -1, -1],
               [-1, -1,  1,  2],
               [ 3,  0, -1, -1],
               [-1, -1,  2,  1]])
        """
        return self._output_edges

    def process(self, input_sequence):
        """
        Returns the output sequence corresponding to a given input sequence. The input sequence and the output sequence are denoted by :math:`\\mathbf{x} = (x_0, x_1, \\ldots, x_{L-1}) \\in \\mathcal{X}^L` and :math:`\\mathbf{y} = (y_0, y_1, \\ldots, y_{L-1}) \\in \\mathcal{Y}^L`, respectively. This method takes into account the current state of the machine.

        **Input:**

        :code:`input_sequence` : 1D-array of :obj:`int`
            The input sequence :math:`\\mathbf{x} \\in \\mathcal{X}^L`. It should be a 1D-array with elements in :math:`\\mathcal{X}`.

        **Output:**

        :code:`output_sequence` : 1D-array of :obj:`int`
            The output sequence :math:`\\mathbf{y} \\in \\mathcal{Y}^L` corresponding to :code:`input_sequence`, taking into account the current state of the machine. It is a 1D-array with elements in :math:`\\mathcal{Y}`.
        """
        output_sequence = np.empty_like(input_sequence, dtype=np.int)
        for t, x in np.ndenumerate(input_sequence):
            y = self._outputs[self._state, x]
            self._state = self._next_states[self._state, x]
            output_sequence[t] = y
        return output_sequence

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

    def forward_backward(self, observed_sequence, metric_function, input_priors=None, initial_state=0, final_state=0):
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

        :code:`initial_state` : :obj:`int`, optional
            The initial state of the machine. It must be an integer in :math:`\\mathcal{S}`. The default value is :code:`0`.

        :code:`final_state` : :obj:`int`, optional
            The final state of the machine. It must be an integer in :math:`\\mathcal{S}`. The default value is :code:`0`.

        **Output:**

        :code:`input_posteriors` : 2D-array of :obj:`float`
            The posterior :term:`pmf` of each input, given the observed sequence, of shape :math:`L \\times |\\mathcal{X}|`. The element in row :math:`t \\in [0 : L)` and column :math:`x \\in \\mathcal{X}` is :math:`p(x_t = x \mid \\mathbf{z})`.
        """
        L, num_states, num_input_symbols = len(observed_sequence), self._num_states, self._num_input_symbols

        if input_priors is None:
            log_input_priors = np.full((L, num_input_symbols), fill_value=0.0)
        else:
            log_input_priors = np.log(input_priors)

        log_gamma = np.full((L, num_states, num_states), fill_value=-np.inf)
        for t, z in enumerate(observed_sequence):
            for x, s0 in np.ndindex(num_input_symbols, num_states):
                y, s1 = self._outputs[s0, x], self._next_states[s0, x]
                log_gamma[t, s0, s1] = log_input_priors[t, x] + metric_function(y, z)

        log_alpha = np.full((L + 1, num_states), fill_value=-np.inf)
        log_alpha[0, initial_state] = 0.0
        for t in range(0, L - 1):
            for s1 in range(num_states):
                log_alpha[t + 1, s1] = logsumexp(log_gamma[t, :, s1] + log_alpha[t, :])

        log_beta = np.full((L + 1, num_states), fill_value=-np.inf)
        log_beta[L, final_state] = 0.0
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


class ConvolutionalCode:
    """
    Binary convolutional code. It is characterized by a *matrix of feedforward polynomials* :math:`P(D)`, of shape :math:`k \\times n`, and (optionally) by a *vector of feedback polynomials* :math:`q(D)`, of length :math:`k`. The element in row :math:`i` and column :math:`j` of :math:`P(D)` is denoted by :math:`p_{i,j}(D)`, and the element in position :math:`i` of :math:`q(D)` is denoted by :math:`q_i(D)`; they are binary polynomials (:class:`BinaryPolynomial`) in :math:`D`. The parameters :math:`k` and :math:`n` are the number of input and output bits per block, respectively.

    The *transfer function matrix* (also known as *transform domain generator matrix*) :math:`G(D)` of the convolutional code, of shape :math:`k \\times n`, is such that the element in row :math:`i` and column :math:`j` is given by

    .. math::
       g_{i,j}(D) = \\frac{p_{i,j}(D)}{q_{i}(D)},

    for :math:`i \\in [0 : k)` and :math:`j \\in [0 : n)`.

    **Decoding methods**

    .. csv-table::
       :header: Method, Identifier, Input type

       Viterbi (hard-decision), :code:`viterbi_hard`, hard
       Viterbi (soft-decision), :code:`viterbi_soft`, soft
       BCJR, :code:`bcjr`, soft

    **Table of convolutional codes**

    The table below lists optimal convolutional codes with parameters :math:`(n,k) = (2,1)` and :math:`(n,k) = (3,1)`, for small values of the overall constraint length :math:`\\nu`. For more details, see :cite:`Lin.Costello.04` (Sec. 12.3).

    =================================  ======================================
     Parameters :math:`(n, k, \\nu)`    Transfer function matrix :math:`G(D)`
    =================================  ======================================
     :math:`(2, 1, 1)`                  :code:`[[0o1, 0o3]]`
     :math:`(2, 1, 2)`                  :code:`[[0o5, 0o7]]`
     :math:`(2, 1, 3)`                  :code:`[[0o13, 0o17]]`
     :math:`(2, 1, 4)`                  :code:`[[0o27, 0o31]]`
     :math:`(2, 1, 5)`                  :code:`[[0o53, 0o75]]`
     :math:`(2, 1, 6)`                  :code:`[[0o117, 0o155]]`
     :math:`(2, 1, 7)`                  :code:`[[0o247, 0o371]]`
     :math:`(2, 1, 8)`                  :code:`[[0o561, 0o753]]`
    =================================  ======================================

    =================================  ======================================
     Parameters :math:`(n, k, \\nu)`    Transfer function matrix :math:`G(D)`
    =================================  ======================================
     :math:`(3, 1, 1)`                  :code:`[[0o1, 0o3, 0o3]]`
     :math:`(3, 1, 2)`                  :code:`[[0o5, 0o7, 0o7]]`
     :math:`(3, 1, 3)`                  :code:`[[0o13, 0o15, 0o17]]`
     :math:`(3, 1, 4)`                  :code:`[[0o25, 0o33, 0o37]]`
     :math:`(3, 1, 5)`                  :code:`[[0o47, 0o53, 0o75]]`
     :math:`(3, 1, 6)`                  :code:`[[0o117, 0o127, 0o155]]`
     :math:`(3, 1, 7)`                  :code:`[[0o255, 0o331, 0o367]]`
     :math:`(3, 1, 8)`                  :code:`[[0o575, 0o623, 0o727]]`
    =================================  ======================================

    References: :cite:`Johannesson.Zigangirov.15`, :cite:`Lin.Costello.04`
    """

    def __init__(self, feedforward_polynomials, feedback_polynomials=None):
        """
        Constructor for the class. It expects the following parameters:

        :code:`feedforward_polynomials` : 2D-array of (:obj:`BinaryPolynomial` or :obj:`int`)
            The matrix of feedforward polynomials :math:`P(D)`, which is a :math:`k \\times n` matrix whose entries are either binary polynomials (:obj:`BinaryPolynomial`) or integers to be converted to the former.

        :code:`feedback_polynomials` : 1D-array of  (:obj:`BinaryPolynomial` or :obj:`int`), optional
            The vector of feedback polynomials :math:`q(D)`, which is a :math:`k`-vector whose entries are either binary polynomials (:obj:`BinaryPolynomial`) or integers to be converted to the former. The default value corresponds to no feedback, that is, :math:`q_i(D) = 1` for all :math:`i \\in [0 : k)`.

        .. rubric:: Examples

        The convolutional code with encoder depicted in the figure below has parameters :math:`(n, k, \\nu) = (2, 1, 6)`; its transfer function matrix is given by

        .. math::

           G(D) =
           \\begin{bmatrix}
              D^6 + D^3 + D^2 + D + 1  &  D^6 + D^5 + D^3 + D^2 + 1
           \\end{bmatrix},

        yielding :code:`feedforward_polynomials = [[0b1001111, 0b1101101]] = [[0o117, 0o155]] = [[79, 109]]`.

        .. image:: figures/cc_2_1_6.png
           :alt: Convolutional encoder example.
           :align: center

        >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o117, 0o155]])
        >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
        (2, 1, 6)

        The convolutional code with encoder depicted in the figure below has parameters :math:`(n, k, \\nu) = (3, 2, 7)`; its transfer function matrix is given by

        .. math::

           G(D) =
           \\begin{bmatrix}
               D^4 + D^3 + 1  &  D^4 + D^2 + D + 1  &  0 \\\\
               0  &  D^3 + D  &  D^3 + D^2 + 1 \\\\
           \\end{bmatrix},

        yielding :code:`feedforward_polynomials = [[0b11001, 0b10111, 0b00000], [0b0000, 0b1010, 0b1101]] = [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]] = [[25, 23, 0], [0, 10, 13]]`.

        .. image:: figures/cc_3_2_7.png
           :alt: Convolutional encoder example.
           :align: center

        >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]])
        >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
        (3, 2, 7)

        The convolutional code with feedback encoder depicted in the figure below has parameters :math:`(n, k, \\nu) = (2, 1, 4)`; its transfer function matrix is given by

        .. math::

           G(D) =
           \\begin{bmatrix}
               1  &  \\dfrac{D^4 + D^3 + 1}{D^4 + D^2 + D + 1}
           \\end{bmatrix},

        yielding :code:`feedforward_polynomials = [[0b10111, 0b11001]] = [[0o27, 0o31]] = [[23, 25]]` and :code:`feedback_polynomials = [0o27]`.

        .. image:: figures/cc_2_1_4_fb.png
           :alt: Convolutional feedback encoder example.
           :align: center

        >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o27, 0o31]], feedback_polynomials=[0o27])
        >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
        (2, 1, 4)
        """
        self._feedforward_polynomials = np.empty_like(feedforward_polynomials, dtype=BinaryPolynomial)
        for (i, j), p in np.ndenumerate(feedforward_polynomials):
            self._feedforward_polynomials[i, j] = BinaryPolynomial(p)

        k, n = self._feedforward_polynomials.shape

        if feedback_polynomials == None:
            self._feedback_polynomials = np.array([BinaryPolynomial(0b1) for _ in range(k)], dtype=np.object)
            self._constructed_from = 'no_feedback_polynomials'
        else:
            self._feedback_polynomials = np.empty_like(feedback_polynomials, dtype=np.object)
            for i, q in np.ndenumerate(feedback_polynomials):
                self._feedback_polynomials[i] = BinaryPolynomial(q)
            self._constructed_from = 'feedback_polynomials'

        nus = np.empty(k, dtype=np.int)
        for i, (ps, q) in enumerate(zip(self._feedforward_polynomials, self._feedback_polynomials)):
            nus[i] = max(np.amax([p.degree for p in ps]), q.degree)

        self._num_input_bits = k
        self._num_output_bits = n
        self._constraint_lengths = nus
        self._overall_constraint_length = np.sum(nus)
        self._memory_order = np.amax(nus)

        self._transfer_function_matrix = np.empty((k, n), dtype=np.object)
        for (i, j), p in np.ndenumerate(feedforward_polynomials):
            q = self._feedback_polynomials[i]
            self._transfer_function_matrix[i, j] = BinaryPolynomialFraction(p) / BinaryPolynomialFraction(q)

        self._setup_finite_state_machine_direct_form()

        cache = np.array([int2binlist(y, width=n) for y in range(2**n)])
        self._metric_function_viterbi_hard = lambda y, z: np.count_nonzero(cache[y] != z)
        self._metric_function_viterbi_soft = lambda y, z: np.dot(cache[y], z)

    def __repr__(self):
        feedforward_polynomials_str = str(np.vectorize(str)(self._feedforward_polynomials).tolist()).replace("'", "")
        args = 'feedforward_polynomials={}'.format(feedforward_polynomials_str)
        if self._constructed_from == 'feedback_polynomials':
            feedback_polynomials_str = str(np.vectorize(str)(self._feedback_polynomials).tolist()).replace("'", "")
            args = '{}, feedback_polynomials={}'.format(args, feedback_polynomials_str)
        return '{}({})'.format(self.__class__.__name__, args)

    def _setup_finite_state_machine_direct_form(self):
        n, k, nu = self._num_output_bits, self._num_input_bits, self._overall_constraint_length

        x_indices = np.concatenate(([0], np.cumsum(self._constraint_lengths + 1)[:-1]))
        s_indices = np.setdiff1d(np.arange(k + nu), x_indices)

        feedforward_taps = []
        for j in range(n):
            taps = np.concatenate([self._feedforward_polynomials[i, j].exponents() + x_indices[i] for i in range(k)])
            feedforward_taps.append(taps)

        feedback_taps = []
        for i in range(k):
            taps = (BinaryPolynomial(0b1) + self._feedback_polynomials[i]).exponents() + x_indices[i]
            feedback_taps.append(taps)

        bits = np.empty(k + nu, dtype=np.int)
        next_states = np.empty((2**nu, 2**k), dtype=np.int)
        outputs = np.empty((2**nu, 2**k), dtype=np.int)

        for s, x in np.ndindex(2**nu, 2**k):
            bits[s_indices] = int2binlist(s, width=nu)
            bits[x_indices] = int2binlist(x, width=k)
            bits[x_indices] ^= [np.bitwise_xor.reduce(bits[feedback_taps[i]]) for i in range(k)]

            next_state_bits = bits[s_indices - 1]
            output_bits = [np.bitwise_xor.reduce(bits[feedforward_taps[j]]) for j in range(n)]

            next_states[s, x] = binlist2int(next_state_bits)
            outputs[s, x] = binlist2int(output_bits)

        self._finite_state_machine =  FiniteStateMachine(next_states=next_states, outputs=outputs)

    def _setup_finite_state_machine_transposed_form(self):
        pass

    @property
    def num_input_bits(self):
        """
        The number of input bits per block, :math:`k`. This property is read-only.
        """
        return self._num_input_bits

    @property
    def num_output_bits(self):
        """
        The number of output bits per block, :math:`n`. This property is read-only.
        """
        return self._num_output_bits

    @property
    def constraint_lengths(self):
        """
        The constraint lengths :math:`\\nu_i` of the code, for :math:`i \\in [0 : k)`. This is a 1D-array of :obj:`int`. It is given by

        .. math::

            \\nu_i = \\max \\{ \\deg p_{i,0}(D), \\deg p_{i,1}(D), \\ldots, \\deg p_{i,n-1}(D), \\deg q_i(D) \\}.

        This property is read-only.
        """
        return self._constraint_lengths

    @property
    def overall_constraint_length(self):
        """
        The overall constraint length :math:`\\nu` of the code. It is given by

        .. math::

            \\nu = \\sum_{0 \\leq i < k} \\nu_i

        This property is read-only.
        """
        return self._overall_constraint_length

    @property
    def memory_order(self):
        """
        The memory order :math:`\\mu` of the code. It is given by

        .. math::

            \\mu = \\max_{0 \\leq i < k} \\nu_i

        This property is read-only.
        """
        return  self._memory_order

    @property
    def feedforward_polynomials(self):
        """
        The matrix of feedforward polynomials :math:`P(D)` of the code. This is a :math:`k \\times n` array of :obj:`BinaryPolynomial`. This property is read-only.
        """
        return self._feedforward_polynomials

    @property
    def feedback_polynomials(self):
        """
        The vector of feedback polynomials :math:`q(D)` of the code. This is a :math:`k`-array of :obj:`BinaryPolynomial`. This property is read-only.
        """
        return self._feedback_polynomials

    @property
    def transfer_function_matrix(self):
        """
        The transfer function matrix :math:`G(D)` of the code. This is a :math:`k \\times n` array of :obj:`BinaryPolynomialFraction`. This property is read-only.
        """
        return self._transfer_function_matrix

    def encode(self, message, initial_state=0, method=None):
        """
        Encodes a given message to its corresponding codeword.

        **Input:**

        :code:`message` : 1D-array of :obj:`int`
            Binary message to be encoded. Its length must be a multiple of :math:`k`.

        :code:`initial_state` : :obj:`int`, optional
            Initial state of the machine. The default value is :code:`0`.

        :code:`method` : :obj:`str`, optional
            Encoding method to be used.

        **Output:**

        :code:`codeword` : 1D-array of :obj:`int`
            Codeword corresponding to :code:`message`. Its length is equal to :math:`(n/k)` times the length of :code:`message`.
        """
        message = np.array(message)
        if method is None:
            method = self._default_encoder()
        encoder = getattr(self, '_encode_' + method)
        codeword = encoder(message)  # TODO: check initial_state...
        return codeword

    def _encode_finite_state_machine(self, message, initial_state=0):
        input_sequence = pack(message, width=self._num_input_bits)
        self._finite_state_machine.state = initial_state
        output_sequence = self._finite_state_machine.process(input_sequence)
        codeword = unpack(output_sequence, width=self._num_output_bits)
        return codeword

    def _default_encoder(self):
        return 'finite_state_machine'

    def decode(self, recvword, method=None):
        """
        Decodes a received word to a message.

        **Input:**

        :code:`recvword` : 1D-array of (:obj:`int` or :obj:`float`)
            Word to be decoded. If using a hard-decision decoding method, then the elements of the array must be bits (integers in :math:`\{ 0, 1 \}`). If using a soft-decision decoding method, then the elements of the array must be soft-bits (floats standing for log-probability ratios, in which positive values represent bit :math:`0` and negative values represent bit :math:`1`). Its length must be a multiple of :math:`n`.

        :code:`method` : :obj:`str`, optional
            Decoding method to be used.

        **Output:**

        :code:`message_hat` : 1D-array of :obj:`int`
            Message decoded from :code:`recvword`. Its length is equal to :math:`(k/n)` times the length of :code:`recvword`.
        """
        recvword = np.array(recvword)
        if method is None:
            method = self._default_decoder(recvword.dtype)
        decoder = getattr(self, '_decode_' + method)
        message_hat = decoder(recvword)
        return message_hat

    def _helper_decode_viterbi(self, recvword, metric_function):
        initial_metrics = np.full(2**self._overall_constraint_length, fill_value=np.inf)
        initial_metrics[0] = 0.0
        observed_sequence = np.reshape(recvword, newshape=(-1, self._num_output_bits))
        input_sequences_hat, final_metrics = self._finite_state_machine.viterbi(observed_sequence, metric_function, initial_metrics)
        input_sequence_hat = input_sequences_hat[:, 0]
        message_hat = unpack(input_sequence_hat, width=self._num_input_bits)
        return message_hat

    @tag(name='Viterbi (hard-decision)', input_type='hard', target='message')
    def _decode_viterbi_hard(self, recvword):
        return self._helper_decode_viterbi(recvword, self._metric_function_viterbi_hard)

    @tag(name='Viterbi (soft)', input_type='soft', target='message')
    def _decode_viterbi_soft(self, recvword):
        return self._helper_decode_viterbi(recvword, self._metric_function_viterbi_soft)

    @tag(name='BCJR', input_type='soft', target='message')
    def _decode_bcjr(self, recvword, SNR=1.0):
        observed = np.reshape(recvword, newshape=(-1, self._num_output_bits))
        cache = (-1)**np.array([int2binlist(y, width=self._num_output_bits) for y in range(2**self._num_output_bits)])
        metric_function = lambda y, z: 4 * SNR * np.dot(cache[y], z)
        input_posteriors = self._finite_state_machine.forward_backward(observed, metric_function=metric_function)
        input_sequence_hat = np.argmax(input_posteriors, axis=1)
        message_hat = unpack(input_sequence_hat, width=self._num_input_bits)
        return message_hat

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'viterbi_hard'
        elif dtype == np.float:
            return 'viterbi_soft'


class TerminatedConvolutionalCode(BlockCode, ConvolutionalCode):
    """
    Terminated binary convolutional code. It is a linear block code (:class:`BlockCode`) obtained by terminating a convolutional code (:class:`ConvolutionalCode`). A total of :math:`h` information blocks (each containing :math:`k` information bits) is encoded. The dimension of the resulting block code is thus :math:`K = kh`; its length depends on the termination mode employed. There are three possible termination modes:

    **Truncated** (or direct). The encoder always starts at state :math:`0`, and its output ends immediately after the last information block. The encoder may not necessarily end in state :math:`0`. The resulting block code will have length :math:`N = nh`.

    **Zero-tail**. The encoder always starts and ends at state :math:`0`. To achieve this, a sequence of :math:`km` zero bits is appended to the information bits, where :math:`m` is the memory order of the convolutional code. The resulting block code will have length :math:`N = n(h + m)`.

    **Tail-biting**. The encoder always starts and ends at the same state. To achieve this, the initial state of the encoder is chosen as a function of the information bits. The resulting block code will have length :math:`N = nh`.

    Only feedforward convolutional codes are supported.

    References: :cite:`Lin.Costello.04`
    """
    def __init__(self, feedforward_polynomials, num_blocks, mode='zero-tail'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`feedforward_polynomials` : 2D-array of (:obj:`BinaryPolynomial` or :obj:`int`)
            The matrix of feedforward polynomials :math:`P(D)` of the convolutional code. For more details, check the documentation in :class:`ConvolutionalCode`.

        :code:`num_blocks` : :obj:`int`
            The number :math:`h` of information blocks.

        :code:`mode` : :obj:`str`, optional
            The termination mode. It must be one of :code:`'truncated'`, :code:`'zero-tail'`, or :code:`'tail-biting'`. The default value is :code:`'zero-tail'`.

        .. rubric:: Examples

        >>> code = komm.TerminatedConvolutionalCode(feedforward_polynomials=[[0b1, 0b11]], num_blocks=3, mode='zero-tail')
        >>> (code.length, code.dimension, code.minimum_distance)
        (8, 3, 3)
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 1]])

        >>> code = komm.TerminatedConvolutionalCode(feedforward_polynomials=[[0b1, 0b11]], num_blocks=3, mode='truncated')
        >>> (code.length, code.dimension, code.minimum_distance)
        (6, 3, 2)
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 0, 1],
               [0, 0, 0, 0, 1, 1]])

        >>> code = komm.TerminatedConvolutionalCode(feedforward_polynomials=[[0b1, 0b11]], num_blocks=3, mode='tail-biting')
        >>> (code.length, code.dimension, code.minimum_distance)
        (6, 3, 3)
        >>> code.generator_matrix
        array([[1, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 0, 1],
               [0, 1, 0, 0, 1, 1]])
        """
        ConvolutionalCode.__init__(self, feedforward_polynomials)

        k, n, m = self._num_input_bits, self._num_output_bits, self.memory_order

        sliced_generator_matrices = np.empty((m + 1, k, n), dtype=np.int)

        for i, j in np.ndindex(k, n):
            sliced_generator_matrices[:, i, j] = self._feedforward_polynomials[i, j].coefficients(width=m + 1)

        base_rows_left = np.concatenate([sliced_generator_matrices[ell, :, :] for ell in range(m + 1)], axis=1)
        base_rows_right = np.zeros((k, n*(num_blocks - m - 1)), dtype=np.int)
        base_rows = np.concatenate([base_rows_left, base_rows_right], axis=1)

        if mode == 'truncated':
            generator_matrix = np.zeros((k*num_blocks, n*num_blocks), dtype=np.int)
            for t in range(num_blocks):
                generator_matrix[k*t : k*(t + 1), :] = np.roll(base_rows, shift=n*t,  axis=1)
                generator_matrix[k*t : k*(t + 1), : n*t] = 0
        elif mode == 'zero-tail':
            generator_matrix = np.zeros((k*num_blocks, n*(num_blocks + m)), dtype=np.int)
            for t in range(num_blocks):
                generator_matrix[k*t : k*(t + 1), n*t : n*t + n*(m + 1)] = base_rows_left
        elif mode == 'tail-biting':
            generator_matrix = np.zeros((k*num_blocks, n*num_blocks), dtype=np.int)
            for t in range(num_blocks):
                generator_matrix[k*t : k*(t + 1), :] = np.roll(base_rows, shift=n*t,  axis=1)
        else:
            raise ValueError("Parameter 'mode' must be in {'truncated', 'zero-tail', 'tail-biting'}")

        BlockCode.__init__(self, generator_matrix=generator_matrix)

        self._mode = mode
        self._num_blocks = num_blocks

        cache = np.array([int2binlist(y, width=n) for y in range(2**n)])
        self._metric_function_viterbi_hard = lambda y, z: np.count_nonzero(cache[y] != z)
        self._metric_function_viterbi_soft = lambda y, z: np.dot(cache[y], z)

    @property
    def num_blocks(self):
        """
        The number :math:`h` of information blocks of the terminated convolutional code.
        """
        return self._num_blocks

    @property
    def mode(self):
        """
        The termination mode of the terminated convolutional code.
        """
        return self._mode

    def _encode_finite_state_machine(self, message):
        if self._mode == 'truncated':
            tail = np.empty(0, dtype=np.int)
            self._finite_state_machine.state = 0
        elif self._mode == 'zero-tail':
            tail = np.zeros(self._num_input_bits*self.memory_order, dtype=np.int)
            self._finite_state_machine.state = 0
        elif self._mode == 'tail-biting':
            tail = np.empty(0, dtype=np.int)
            self._finite_state_machine.state = binlist2int(message[-self._num_input_bits*self.memory_order :])
        input_sequence = pack(np.concatenate([message, tail]), width=self._num_input_bits)
        output_sequence = self._finite_state_machine.process(input_sequence)
        codeword = unpack(output_sequence, width=self._num_output_bits)
        return codeword

    def _default_encoder(self):
        return 'finite_state_machine'

    def _helper_decode_viterbi(self, recvword, metric_function):
        if self._mode in ['truncated', 'zero-tail']:
            initial_metrics = np.full(2**self._overall_constraint_length, fill_value=np.inf)
            initial_metrics[0] = 0.0
        elif self._mode == 'tail-biting':
            initial_metrics = np.zeros(2**self._overall_constraint_length, dtype=np.float)

        observed_sequence = np.reshape(recvword, newshape=(-1, self._num_output_bits))
        input_sequences_hat, final_metrics = self._finite_state_machine.viterbi(observed_sequence, metric_function, initial_metrics)

        if self._mode == 'truncated':
            final_state_hat = np.argmin(final_metrics)
            input_sequence_hat = input_sequences_hat[:, final_state_hat]
        elif self._mode == 'zero-tail':
            input_sequence_hat = input_sequences_hat[:, 0][: -self._num_input_bits*self._memory_order]
        elif self._mode == 'tail-biting':
            pass

        message_hat = unpack(input_sequence_hat, width=self._num_input_bits)
        return message_hat

    @tag(name='Viterbi (hard-decision)', input_type='hard', target='message')
    def _decode_viterbi_hard(self, recvword):
        return self._helper_decode_viterbi(recvword, self._metric_function_viterbi_hard)

    @tag(name='Viterbi (soft)', input_type='soft', target='message')
    def _decode_viterbi_soft(self, recvword):
        return self._helper_decode_viterbi(recvword, self._metric_function_viterbi_soft)

    @tag(name='BCJR', input_type='soft', target='message')
    def _decode_bcjr(self, recvword, SNR=1.0):
        observed = np.reshape(recvword, newshape=(-1, self._num_output_bits))
        cache = (-1)**np.array([int2binlist(y, width=self._num_output_bits) for y in range(2**self._num_output_bits)])
        metric_function = lambda y, z: 4 * SNR * np.dot(cache[y], z)
        input_posteriors = self._finite_state_machine.forward_backward(observed, metric_function=metric_function)
        input_sequence_hat = np.argmax(input_posteriors, axis=1)
        message_hat = unpack(input_sequence_hat, width=self._num_input_bits)
        return message_hat

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'viterbi_hard'
        elif dtype == np.float:
            return 'viterbi_soft'
