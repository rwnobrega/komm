import numpy as np

from scipy.special import logsumexp

from ._algebra import \
    BinaryPolynomial, BinaryPolynomialFraction

from .util import \
    int2binlist, binlist2int, pack, unpack

from ._aux import tag

__all__ = ['FiniteStateMachine', 'ConvolutionalCode',
           'ConvolutionalEncoder', 'ConvolutionalDecoder']


class FiniteStateMachine:
    """
    Finite-state machine (Mealy machine). It is defined by a *set of states* :math:`\\mathcal{S}`, an *input alphabet* :math:`\\mathcal{X}`, an *output alphabet* :math:`\\mathcal{Y}`, and a *transition function* :math:`T : \\mathcal{S} \\times \\mathcal{X} \\to \\mathcal{S} \\times \\mathcal{Y}`. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are always taken as :math:`\\mathcal{S} = \\{ 0, 1, \ldots, |\\mathcal{S}| - 1 \\}`, :math:`\\mathcal{X} = \\{ 0, 1, \ldots, |\\mathcal{X}| - 1 \\}`, and :math:`\\mathcal{Y} = \\{ 0, 1, \ldots, |\\mathcal{Y}| - 1 \\}`, respectively.

    For example, consider the finite-state machine whose state diagram depicted in the figure below.

    .. image:: figures/mealy.png
       :alt: Finite-state machine (Mealy machine) example.
       :align: center

    It has set of states :math:`\\mathcal{S} = \\{ 0, 1, 2, 3 \\}`, input alphabet :math:`\\mathcal{X} = \\{ 0, 1 \\}`, output alphabet :math:`\\mathcal{Y} = \\{ 0, 1, 2, 3 \\}`, and transition function :math:`T` given by the table below.

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
            The posterior :term:`pmf` of each input, given the observed sequence, of shape :math:`L \\times |\\mathcal{X}|`. The element in row :math:`t \\in [0 : L)` and column :math:`x \\in \\mathcal{X}` is :math:`p(x_t = x \mid \\mathbf{z})`.
        """
        L, num_states, num_input_symbols = len(observed_sequence), self._num_states, self._num_input_symbols

        if input_priors is None:
            log_input_priors = np.full((L, num_input_symbols), fill_value=0.0)
        else:
            with np.errstate(divide='ignore'):
                log_input_priors = np.log(input_priors)

        log_gamma = np.full((L, num_states, num_states), fill_value=-np.inf)
        for t, z in enumerate(observed_sequence):
            for x, s0 in np.ndindex(num_input_symbols, num_states):
                y, s1 = self._outputs[s0, x], self._next_states[s0, x]
                log_gamma[t, s0, s1] = log_input_priors[t, x] + metric_function(y, z)


        log_alpha = np.full((L + 1, num_states), fill_value=-np.inf)
        if initial_state_distribution is None:
            log_alpha[0, :] = 0.0
        else:
            with np.errstate(divide='ignore'):
                log_alpha[0, :] = np.log(initial_state_distribution)

        for t in range(0, L - 1):
            for s1 in range(num_states):
                log_alpha[t + 1, s1] = logsumexp(log_gamma[t, :, s1] + log_alpha[t, :])

        log_beta = np.full((L + 1, num_states), fill_value=-np.inf)
        if final_state_distribution is None:
            log_beta[L, :] = 0.0
        else:
            with np.errstate(divide='ignore'):
                log_beta[L, :] = np.log(final_state_distribution)

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


class ConvolutionalEncoder:
    """
    Convolutional encoder.

    **Input:**

    :code:`message` : 1D-array of :obj:`int`
        Binary message to be encoded. Its length must be a multiple of :math:`k`.

    **Output:**

    :code:`codeword` : 1D-array of :obj:`int`
        Codeword corresponding to :code:`message`. Its length is equal to :math:`(n/k)` times the length of :code:`message`.

    .. rubric:: Examples

    >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
    >>> convolutional_encoder = komm.ConvolutionalEncoder(convolutional_code)
    >>> convolutional_encoder([1, 0, 1, 1, 1, 0, 1, 1, 0, 0])
    array([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    """
    def __init__(self, convolutional_code, initial_state=0, method='finite_state_machine'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`convolutional_code` : :class:`komm.ConvolutionalCode`
            The convolutional code.

        :code:`initial_state` : :obj:`int`, optional
            Initial state of the machine. The default value is :code:`0`.

        :code:`method` : :obj:`str`, optional
            Encoding method to be used.
        """
        self._convolutional_code = convolutional_code
        self._state = int(initial_state)
        self._method = method

        try:
            self._encoder = getattr(self, '_encode_' + method)
        except AttributeError:
            raise ValueError("Unsupported encoding method")

    def __call__(self, inp):
        return self._encoder(np.array(inp))

    def _encode_finite_state_machine(self, message):
        n, k = self._convolutional_code._num_output_bits, self._convolutional_code._num_input_bits
        fsm = self._convolutional_code._finite_state_machine
        input_sequence = pack(message, width=k)
        output_sequence, self._state = fsm.process(input_sequence, self._state)
        codeword = unpack(output_sequence, width=n)
        return codeword


class ConvolutionalDecoder:
    """
    Convolutional decoder.

    **Input:**

    :code:`recvword` : 1D-array of (:obj:`int` or :obj:`float`)
        Word to be decoded. If using a hard-decision decoding method, then the elements of the array must be bits (integers in :math:`\{ 0, 1 \}`). If using a soft-decision decoding method, then the elements of the array must be soft-bits (floats standing for log-probability ratios, in which positive values represent bit :math:`0` and negative values represent bit :math:`1`). Its length must be a multiple of :math:`n`.

    **Output:**

    :code:`message_hat` : 1D-array of :obj:`int`
        Message decoded from :code:`recvword`. Its length is equal to :math:`(k/n)` times the length of :code:`recvword`.

    .. rubric:: Examples

    >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
    >>> convolutional_decoder = komm.ConvolutionalDecoder(convolutional_code)
    >>> convolutional_decoder([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0])
    """
    def __init__(self, convolutional_code, initial_state=0, channel_snr=1.0, input_type='hard', output_type='hard', method='viterbi'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`convolutional_code` : :class:`komm.ConvolutionalCode`
            The convolutional code.

        :code:`initial_state` : :obj:`int`, optional
            Initial state of the machine. The default value is :code:`0`.

        :code:`method` : :obj:`str`, optional
            Decoding method to be used.
        """
        self._convolutional_code = convolutional_code
        self._state = int(initial_state)
        self._channel_snr = float(channel_snr)
        self._method = method

        try:
            self._decoder = getattr(self, '_decode_{}_{}_{}'.format(method, input_type, output_type))
        except AttributeError:
            raise ValueError("Unsupported decoding method")

        n = self._convolutional_code._num_output_bits
        cache_bit = np.array([int2binlist(y, width=n) for y in range(2**n)])
        self._metric_function_viterbi_hard = lambda y, z: np.count_nonzero(cache_bit[y] != z)
        self._metric_function_viterbi_soft = lambda y, z: np.dot(cache_bit[y], z)
        cache_polar = (-1)**cache_bit
        self._metric_function_bcjr = lambda y, z: 2.0 * self._channel_snr * np.dot(cache_polar[y], z)

    def __call__(self, inp):
        return self._decoder(np.array(inp))

    def _helper_decode_viterbi(self, recvword, input_type):
        code = self._convolutional_code
        n, k = code._num_output_bits, code._num_input_bits
        num_states = code._finite_state_machine._num_states
        initial_metrics = np.full(num_states, fill_value=np.inf)
        initial_metrics[0] = 0.0
        input_sequences_hat, final_metrics = code._finite_state_machine.viterbi(
            observed_sequence=np.reshape(recvword, newshape=(-1, n)),
            metric_function=getattr(self, '_metric_function_viterbi_' + input_type),
            initial_metrics=initial_metrics)
        input_sequence_hat = input_sequences_hat[:, 0]
        message_hat = unpack(input_sequence_hat, width=k)
        return message_hat

    @tag(name='Viterbi', input_type='hard', output_type='hard', target='message')
    def _decode_viterbi_hard_hard(self, recvword):
        return self._helper_decode_viterbi(recvword, input_type='hard')

    @tag(name='Viterbi', input_type='soft', output_type='hard', target='message')
    def _decode_viterbi_soft_hard(self, recvword):
        return self._helper_decode_viterbi(recvword, input_type='soft')

    def _helper_decode_bcjr(self, recvword, output_type):
        code = self._convolutional_code
        n, k, m = code._num_output_bits, code._num_input_bits, code._memory_order
        num_states = code._finite_state_machine._num_states
        input_posteriors = code._finite_state_machine.forward_backward(
            observed_sequence=np.reshape(recvword, newshape=(-1, n)),
            metric_function=lambda y, z: self._metric_function_bcjr(y, z),
            initial_state_distribution=np.eye(1, num_states, 0),
            final_state_distribution=np.eye(1, num_states, 0))
        input_posteriors = input_posteriors[:-m]
        if output_type == 'soft':
            return np.log(input_posteriors[:,0] / input_posteriors[:,1])
        elif output_type == 'hard':
            input_sequence_hat = np.argmax(input_posteriors, axis=1)
            return unpack(input_sequence_hat, width=k)

    @tag(name='BCJR', input_type='soft', output_type='hard', target='message')
    def _decode_bcjr_soft_hard(self, recvword, output_type='hard'):
        return self._helper_decode_bcjr(recvword, output_type='hard')

    @tag(name='BCJR', input_type='soft', output_type='soft', target='message')
    def _decode_bcjr_soft_soft(self, recvword, output_type='hard', SNR=1.0):
        return self._helper_decode_bcjr(recvword, output_type='soft')
