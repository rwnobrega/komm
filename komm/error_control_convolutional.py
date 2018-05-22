import numpy as np

from .algebra import \
    BinaryPolynomial, BinaryPolynomialFraction

from .util import \
    int2binlist, binlist2int, pack, unpack, hamming_distance_16, tag

__all__ = ['FiniteStateMachine', 'ConvolutionalCode']


class FiniteStateMachine:
    """
    Finite state machine (Mealy machine). It is defined by a *set of states* :math:`\\mathcal{S}`, a *start state* :math:`s_\\mathrm{i} \\in \\mathcal{S}`, an *input alphabet* :math:`\\mathcal{X}`, an *output alphabet* :math:`\\mathcal{Y}`, a *state transition function* :math:`T : \\mathcal{S} \\times \\mathcal{X} \\to \\mathcal{S}`, and an *output function* :math:`T : \\mathcal{S} \\times \\mathcal{X} \\to \\mathcal{Y}`. Here, for simplicity, the set of states, the input alphabet, and the output alphabet are always taken as :math:`\\mathcal{S} = \\{ 0, 1, \ldots, |\\mathcal{S}| - 1 \\}`, :math:`\\mathcal{X} = \\{ 0, 1, \ldots, |\\mathcal{X}| - 1 \\}`, and :math:`\\mathcal{Y} = \\{ 0, 1, \ldots, |\\mathcal{Y}| - 1 \\}`, respectively.

    For example ...

    >>> fsm = komm.FiniteStateMachine(next_states=[[0, 1], [2, 3], [0, 1], [2, 3]], outputs=[[0, 3], [1, 2], [3, 0], [2, 1]])
    >>> fsm.process([1, 1, 0, 1, 0, 0])
        array([3, 2, 2, 0, 1, 3])
    """
    def __init__(self, next_states, outputs, start_state=0):
        """
        Soon.
        """
        self._next_states = np.array(next_states, dtype=np.int)
        self._outputs = np.array(outputs, dtype=np.int)
        self._num_states, self._num_input_symbols = self._next_states.shape
        self._num_output_symbols = np.max(self._outputs)
        self._state = start_state

        self._input_edges = np.full((self._num_states, self._num_states), fill_value=-1)
        for state_from in range(self._num_states):
            for (i, state_to) in enumerate(self._next_states[state_from, :]):
                self._input_edges[state_from][state_to] = i

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
        The number of states of the finite-state machine. This property is read-only.
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

    def input_edges(self):
        """
        """
        return self._input_edges

    def process(self, input_sequence):
        """
        Returns the output sequence corresponding to a given input sequence. This takes into account the current state of the machine.
        """
        output_sequence = np.empty_like(input_sequence, dtype=np.int)
        for t, x in enumerate(input_sequence):
            y = self._outputs[self._state, x]
            self._state = self._next_states[self._state, x]
            output_sequence[t] = y
        return output_sequence

    def viterbi(self, observed_sequence, metric_fun, start_state=0):   #&* final-state (None or int)
        """
        Applies the Viterbi algorithm on a given observed sequence.

        metric_fun: :math:`\\mathcal{Y} \\times \\mathcal{Z} \\to \\mathbb{R}`
        """
        L = len(observed_sequence)
        choices = np.empty((self._num_states, L), dtype=np.int)
        metrics = np.full((self._num_states, L + 1), fill_value=np.inf)
        metrics[start_state, 0] = 0
        for (t, z) in enumerate(observed_sequence):
            for s0 in range(self._num_states):
                for (s1, y) in zip(self._next_states[s0], self._outputs[s0]):
                    candidate_metrics = metrics[s0, t] + metric_fun(y, z)
                    if candidate_metrics < metrics[s1, t + 1]:
                        metrics[s1, t + 1] = candidate_metrics
                        choices[s1, t] = s0

        # Backtrack
        s1 = 0  #@$% final_state
        input_sequence_hat = np.empty(L, dtype=np.int)
        for t in reversed(range(L)):
            s0 = choices[s1, t]
            input_sequence_hat[t] = self._input_edges[s0, s1]
            s1 = s0

        return input_sequence_hat


class ConvolutionalCode:
    """
    Binary convolutional code. It is characterized by its *generator matrix* :math:`G(D)`, a :math:`k \\times n` matrix whose elements are binary polynomial fractions in :math:`D`. The parameters :math:`k` and :math:`n` are the number of input and output bits per block, respectively. For example, the convolutional code with encoder depicted in the figure below has parameters :math:`(n, k) = (2, 1)`; its generator matrix is given by

    .. math::

        G(D) =
        \\begin{bmatrix}
            D^6 + D^3 + D^2 + D + 1  &  D^6 + D^5 + D^3 + D^2 + 1
        \\end{bmatrix},

    whose integer representation is :code:`[[0b1001111, 0b1101101]] = [[0o117, 0o155]] = [[79, 109]]`.

    .. image:: figures/cc_2_1_6.png
       :alt: Convolutional encoder example.
       :align: center

    As another example, the convolutional code with encoder depicted in the figure below has parameters :math:`(n, k) = (3, 2)`; its generator matrix is given by

    .. math::

        G(D) =
        \\begin{bmatrix}
            D^4 + D^3 + 1  &  D^4 + D^2 + D + 1  &  0 \\\\
            0  &  D^3 + D  &  D^3 + D^2 + 1 \\\\
        \\end{bmatrix},

    whose integer representation is :code:`[[0b11001, 0b10111, 0b00000], [0b0000, 0b1010, 0b1101]] = [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]] = [[25, 23, 0], [0, 10, 13]]`.

    .. image:: figures/cc_3_2_7.png
       :alt: Convolutional encoder example.
       :align: center

    The table below lists optimal convolutional codes with parameters :math:`(n,k) = (2,1)` and :math:`(n,k) = (3,1)`, for small values of the overall constraint length :math:`\\nu`. For more details, see :cite:`Lin.Costello.04` (Sec. 12.3).

    =================================  =================================
     Parameters :math:`(n, k, \\nu)`    Generator matrix :math:`G(D)`
    =================================  =================================
     :math:`(2, 1, 1)`                  :code:`[[0o1, 0o3]]`
     :math:`(2, 1, 2)`                  :code:`[[0o5, 0o7]]`
     :math:`(2, 1, 3)`                  :code:`[[0o13, 0o17]]`
     :math:`(2, 1, 4)`                  :code:`[[0o27, 0o31]]`
     :math:`(2, 1, 5)`                  :code:`[[0o53, 0o75]]`
     :math:`(2, 1, 6)`                  :code:`[[0o117, 0o155]]`
     :math:`(2, 1, 7)`                  :code:`[[0o247, 0o371]]`
     :math:`(2, 1, 8)`                  :code:`[[0o561, 0o753]]`
    =================================  =================================

    =================================  =================================
     Parameters :math:`(n, k, \\nu)`    Generator matrix :math:`G(D)`
    =================================  =================================
     :math:`(3, 1, 1)`                  :code:`[[0o1, 0o3, 0o3]]`
     :math:`(3, 1, 2)`                  :code:`[[0o5, 0o7, 0o7]]`
     :math:`(3, 1, 3)`                  :code:`[[0o13, 0o15, 0o17]]`
     :math:`(3, 1, 4)`                  :code:`[[0o25, 0o33, 0o37]]`
     :math:`(3, 1, 5)`                  :code:`[[0o47, 0o53, 0o75]]`
     :math:`(3, 1, 6)`                  :code:`[[0o117, 0o127, 0o155]]`
     :math:`(3, 1, 7)`                  :code:`[[0o255, 0o331, 0o367]]`
     :math:`(3, 1, 8)`                  :code:`[[0o575, 0o623, 0o727]]`
    =================================  =================================

    References: :cite:`Johannesson.Zigangirov.15`, :cite:`Lin.Costello.04`
    """

    def __init__(self, feedforward_polynomials, feedback_polynomials=None):
        """
        Constructor for the class. It expects the following parameters:

        :code:`feedforward_polynomials` : 2D-array of (:obj:`BinaryPolynomial` or :obj:`int`)
            The matrix of feedforward polynomials, which is a :math:`k \\times n` array whose entries are either binary polynomials (:obj:`BinaryPolynomial`) or integers to be converted to the former.

        :code:`feedback_polynomials` : 1D-array of  (:obj:`BinaryPolynomial` or :obj:`int`), optional
            The vector of feedback polynomials, which is a :math:`k` array whose entries are either binary polynomials (:obj:`BinaryPolynomial`) or integers to be converted to the former. The default value is :code:`None` (no feedback).

        .. rubric:: Examples

        >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o117, 0o155]])
        >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
        (2, 1, 6)

        >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]])
        >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
        (3, 2, 7)

        >>> feedforward_polynomials = [[0b11, 0b0, 0b1001], [0b101, 0b11, 0b10]]
        >>> feedback_polynomials = [0b111, 0b101]
        code = komm.ConvolutionalCode(feedforward_polynomials, feedback_polynomials)
        >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
        (3, 2, 5)
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
            nus[i] = max(np.max([p.degree for p in ps]), q.degree)

        self._num_input_bits = k
        self._num_output_bits = n
        self._constraint_lengths = nus
        self._overall_constraint_length = np.sum(nus)
        self._memory_order = np.max(nus)

        self._generator_matrix = np.empty((k, n), dtype=np.object)
        for (i, j), p in np.ndenumerate(feedforward_polynomials):
            q = self._feedback_polynomials[i]
            self._generator_matrix[i, j] = BinaryPolynomialFraction(p) / BinaryPolynomialFraction(q)

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

            \\nu_i = \\max \\{ \\deg p_{i,0}(D), \\deg p_{i,1}(D), \\ldots, \\deg p_{i,n-1}(D), \\deg q_i(D) \\},

        where :math:`p_{i,j}(D)` is the element in position :math:`(i, j)` of :math:`P(D)`, and :math:`q_{i}(D)` is the element in position :math:`i` of :math:`Q(D)`, for :math:`i \\in [0 : k)` and :math:`j \\in [0 : n)`. This property is read-only.
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
        The vector of feedback polynomials :math:`Q(D)` of the code. This is a :math:`k` array of :obj:`BinaryPolynomial`. This property is read-only.
        """
        return self._feedback_polynomials

    @property
    def generator_matrix(self):
        """
        The generator matrix :math:`G(D)` of the code. This is a :math:`k \\times n` array of :obj:`BinaryPolynomialFraction`. The element in position :math:`(i, j)` of :math:`G(D)` is given by

        .. math::

            g_{i,j}(D) = \\frac{p_{i,j}(D)}{q_{i}(D)},

        for :math:`i \in [0 : k)` and :math:`j \in [0 : n)`, where :math:`p_{i,j}(D)` is the element in position  :math:`(i, j)` of :math:`P(D)`, and :math:`q_{i}(D)` is the element in position :math:`i` of :math:`Q(D)`. This property is read-only.
        """
        return self._generator_matrix

    def encode(self, message, initial_state=0, method=None):
        """
        Encodes a given message to its corresponding codeword.

        **Input:**

        :code:`message` : 1D-array of :obj:`int`
            Binary message to be encoded. It may be of any length.

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
            Word to be decoded. If using a hard-decision decoding method, then the elements of the array must be bits (integers in :math:`\{ 0, 1 \}`). If using a soft-decision decoding method, then the elements of the array must be soft-bits (floats standing for log-probability ratios, in which positive values represent bit :math:`0` and negative values represent bit :math:`1`). It may be of any length.

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

    @tag(name='Viterbi (hard-decision)', input_type='hard', target='message')
    def _decode_viterbi_hard(self, recvword):
        observed = pack(recvword, width=self._num_output_bits)
        input_sequence_hat = self._finite_state_machine.viterbi(observed, metric_fun=hamming_distance_16)
        message_hat = unpack(input_sequence_hat, width=self._num_input_bits)
        return message_hat

    @tag(name='Viterbi (soft)', input_type='soft', target='message')
    def _decode_viterbi_soft(self, recvword):
        observed = np.reshape(recvword, newshape=(-1, self._num_output_bits))
        metric_fun = lambda y, z: np.dot(np.array(int2binlist(y, width=self._num_output_bits)), z)
        input_sequence_hat = self._finite_state_machine.viterbi(observed, metric_fun=metric_fun)
        message_hat = unpack(input_sequence_hat, width=self._num_input_bits)
        return message_hat

    def _decode_bcjr():
        pass

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'viterbi_hard'
        elif dtype == np.float:
            return 'viterbi_soft'
