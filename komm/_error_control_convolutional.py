import numpy as np

from ._algebra import \
    BinaryPolynomial, BinaryPolynomialFraction

from ._finite_state_machine import \
    FiniteStateMachine

from ._util import \
    int2binlist, binlist2int, pack, unpack


__all__ = ['ConvolutionalCode', 'ConvolutionalStreamEncoder', 'ConvolutionalStreamDecoder']


class ConvolutionalCode:
    """
    Binary convolutional code. It is characterized by a *matrix of feedforward polynomials* :math:`P(D)`, of shape :math:`k \\times n`, and (optionally) by a *vector of feedback polynomials* :math:`q(D)`, of length :math:`k`. The element in row :math:`i` and column :math:`j` of :math:`P(D)` is denoted by :math:`p_{i,j}(D)`, and the element in position :math:`i` of :math:`q(D)` is denoted by :math:`q_i(D)`; they are binary polynomials (:class:`BinaryPolynomial`) in :math:`D`. The parameters :math:`k` and :math:`n` are the number of input and output bits per block, respectively.

    The *transfer function matrix* (also known as *transform-domain generator matrix*) :math:`G(D)` of the convolutional code, of shape :math:`k \\times n`, is such that the element in row :math:`i` and column :math:`j` is given by

    .. math::
       g_{i,j}(D) = \\frac{p_{i,j}(D)}{q_{i}(D)},

    for :math:`i \\in [0 : k)` and :math:`j \\in [0 : n)`.

    .. rubric:: Constraint lenghts and related parameters

    The *constraint lengths* of the code are defined by

    .. math::
       \\nu_i = \\max \\{ \\deg p_{i,0}(D), \\deg p_{i,1}(D), \\ldots, \\deg p_{i,n-1}(D), \\deg q_i(D) \\},

    for :math:`i \\in [0 : k)`.

    The *overall constraint length* of the code is defined by

    .. math::
       \\nu = \\sum_{0 \\leq i < k} \\nu_i.

    The *memory order* of the code is defined by

    .. math::
        \\mu = \\max_{0 \\leq i < k} \\nu_i.

    .. rubric:: Space-state representation

    A convolutional code may also be described via the *space-state representation*. Let :math:`\\mathbf{u}_t = (u_t^{(0)}, u_t^{(1)}, \\ldots, u_t^{(k-1)})` be the input block, :math:`\\mathbf{v}_t = (v_t^{(0)}, v_t^{(1)}, \\ldots, v_t^{(n-1)})` be the output block, and :math:`\\mathbf{s}_t = (s_t^{(0)}, s_t^{(1)}, \\ldots, s_t^{(\\nu-1)})` be the state, all defined at time instant :math:`t`. Then,

    .. math::
       \\mathbf{s}_{t+1} & = \\mathbf{s}_t A + \\mathbf{u}_t B, \\\\
       \\mathbf{v}_{t} & = \\mathbf{s}_t C + \\mathbf{u}_t D,

    where :math:`A` is the :math:`\\nu \\times \\nu` *state matrix*, :math:`B` is the :math:`k \\times \\nu` *control matrix*, :math:`C` is the :math:`\\nu \\times n` *observation matrix*, and :math:`D` is the :math:`k \\times n` *transition matrix*.

    .. rubric:: Table of convolutional code

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

    References: :cite:`Johannesson.Zigangirov.15`, :cite:`Lin.Costello.04`, :cite:`Weiss.01`
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

        if feedback_polynomials is None:
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
        self._setup_space_state_representation()

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
            bits[x_indices] ^= [np.count_nonzero(bits[feedback_taps[i]]) % 2 for i in range(k)]

            next_state_bits = bits[s_indices - 1]
            output_bits = [np.count_nonzero(bits[feedforward_taps[j]]) % 2 for j in range(n)]

            next_states[s, x] = binlist2int(next_state_bits)
            outputs[s, x] = binlist2int(output_bits)

        self._finite_state_machine = FiniteStateMachine(next_states=next_states, outputs=outputs)

    def _setup_finite_state_machine_transposed_form(self):
        pass

    def _setup_space_state_representation(self):
        k, n, nu = self._num_input_bits, self._num_output_bits, self._overall_constraint_length

        self._state_matrix = np.empty((nu, nu), dtype=np.int)
        self._observation_matrix = np.empty((nu, n), dtype=np.int)
        for i in range(nu):
            s0 = 2**i
            s1 = self._finite_state_machine.next_states[s0, 0]
            y = self._finite_state_machine.outputs[s0, 0]
            self._state_matrix[i, :] = int2binlist(s1, width=nu)
            self._observation_matrix[i, :] = int2binlist(y, width=n)

        self._control_matrix = np.empty((k, nu), dtype=np.int)
        self._transition_matrix = np.empty((k, n), dtype=np.int)
        for i in range(k):
            x = 2**i
            s1 = self._finite_state_machine.next_states[0, x]
            y = self._finite_state_machine.outputs[0, x]
            self._control_matrix[i, :] = int2binlist(s1, width=nu)
            self._transition_matrix[i, :] = int2binlist(y, width=n)

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
        The constraint lengths :math:`\\nu_i` of the code, for :math:`i \\in [0 : k)`. This is a 1D-array of :obj:`int`. This property is read-only.
        """
        return self._constraint_lengths

    @property
    def overall_constraint_length(self):
        """
        The overall constraint length :math:`\\nu` of the code. This property is read-only.
        """
        return self._overall_constraint_length

    @property
    def memory_order(self):
        """
        The memory order :math:`\\mu` of the code. This property is read-only.
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

    @property
    def finite_state_machine(self):
        """
        The finite-state machine of the code.
        """
        return self._finite_state_machine

    @property
    def state_matrix(self):
        """
        The state matrix :math:`A` of the state-space representation. This is a :math:`\\nu \\times \\nu` array of integers in :math:`\\{ 0, 1 \\}`. This property is read-only.
        """
        return self._state_matrix

    @property
    def control_matrix(self):
        """
        The control matrix :math:`B` of the state-space representation. This is a :math:`k \\times \\nu` array of integers in :math:`\\{ 0, 1 \\}`. This property is read-only.
        """
        return self._control_matrix

    @property
    def observation_matrix(self):
        """
        The observation matrix :math:`C` of the state-space representation. This is a :math:`\\nu \\times n` array of integers in :math:`\\{ 0, 1 \\}`. This property is read-only.
        """
        return self._observation_matrix

    @property
    def transition_matrix(self):
        """
        The transition matrix :math:`D` of the state-space representation. This is a :math:`k \\times n` array of integers in :math:`\\{ 0, 1 \\}`. This property is read-only.
        """
        return self._transition_matrix


class ConvolutionalStreamEncoder:
    """
    Convolutional stream encoder. Encode a bit stream using a given convolutional code (:class:`ConvolutionalCode`). The internal state of the encoder is maintained across each call.

    .. rubric:: Examples

    >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
    >>> convolutional_encoder = komm.ConvolutionalStreamEncoder(convolutional_code)
    >>> convolutional_encoder([1, 1, 1, 1])
    array([1, 1, 0, 1, 1, 0, 1, 0])
    >>> convolutional_encoder([1, 1, 1, 1])
    array([1, 0, 1, 0, 1, 0, 1, 0])
    """
    def __init__(self, convolutional_code, initial_state=0):
        """
        Constructor for the class. It expects the following parameters:

        :code:`convolutional_code` : :class:`ConvolutionalCode`
            The convolutional code.

        :code:`initial_state` : :obj:`int`, optional
            Initial state of the encoder. The default value is :code:`0`.
        """
        self._convolutional_code = convolutional_code
        self._state = int(initial_state)

    def __call__(self, inp):
        n, k = self._convolutional_code.num_output_bits, self._convolutional_code.num_input_bits

        output_sequence, self._state = self._convolutional_code.finite_state_machine.process(
            input_sequence=pack(inp, width=k),
            initial_state=self._state)

        outp = unpack(output_sequence, width=n)
        return outp


class ConvolutionalStreamDecoder:
    """
    Convolutional stream decoder using Viterbi algorithm. Decode a (hard or soft) bit stream given a convolutional code (:class:`ConvolutionalCode`), assuming a traceback length (path memory) of :math:`\\tau`. At time :math:`t`, the decoder chooses the path survivor with best metric at time :math:`t - \\tau` and outputs the corresponding information bits. The output stream has a delay equal to :math:`k \\tau`, where :math:`k` is the number of input bits of the convolutional code. As a rule of thumb, the traceback length is choosen as :math:`\\tau = 5\\mu`, where :math:`\\mu` is the memory order of the convolutional code.

    .. rubric:: Examples

    >>> convolutional_code = komm.ConvolutionalCode([[0o7, 0o5]])
    >>> convolutional_decoder = komm.ConvolutionalStreamDecoder(convolutional_code, traceback_length=10)
    >>> convolutional_decoder([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> convolutional_decoder(np.zeros(2*10, dtype=np.int))
    array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0])
    """
    def __init__(self, convolutional_code, traceback_length, initial_state=0, input_type='hard'):
        """
        Constructor for the class. It expects the following parameters:

        :code:`convolutional_code` : :class:`ConvolutionalCode`
            The convolutional code.

        :code:`traceback_length` : :obj:`int`
            The traceback length (path memory) :math:`\\tau` of the decoder.

        :code:`initial_state` : :obj:`int`, optional
            Initial state of the encoder. The default value is :code:`0`.
        """
        self._convolutional_code = convolutional_code
        self._traceback_length = int(traceback_length)
        self._initial_state = int(initial_state)
        self._input_type = input_type

        n = convolutional_code.num_output_bits
        num_states = convolutional_code.finite_state_machine.num_states

        self._memory = {}
        self._memory['metrics'] = np.full((num_states, traceback_length + 1), fill_value=np.inf)
        self._memory['metrics'][initial_state, -1] = 0.0
        self._memory['paths'] = np.zeros((num_states, traceback_length + 1), dtype=np.int)

        cache_bit = np.array([int2binlist(y, width=n) for y in range(2**n)])
        self._metric_function_hard = lambda y, z: np.count_nonzero(cache_bit[y] != z)
        self._metric_function_soft = lambda y, z: np.dot(cache_bit[y], z)

    def __call__(self, inp):
        n, k = self._convolutional_code.num_output_bits, self._convolutional_code.num_input_bits

        input_sequence_hat = self._convolutional_code.finite_state_machine.viterbi_streaming(
            observed_sequence=np.reshape(inp, newshape=(-1, n)),
            metric_function=getattr(self, '_metric_function_' + self._input_type),
            memory=self._memory)

        outp = unpack(input_sequence_hat, width=k)
        return outp
