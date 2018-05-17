import numpy as np

from .algebra import \
    BinaryPolynomial

from .util import \
    binary_iterator, binlist2int, tag

__all__ = ['ConvolutionalCode']


class ConvolutionalCode:
    """
    Binary convolutional code. It is characterized by its *(polynomial) generator matrix* :math:`G(D)`, a :math:`k \\times n` matrix whose elements are binary polynomials in :math:`D`. The parameters :math:`k` and :math:`n` are the number of input and output bits per block, respectively. For example, the convolutional code with encoder depicted in the figure below has parameters :math:`(n, k) = (2, 1)`; its generator matrix is given by

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

    .. rubric:: Examples

    >>> code = komm.ConvolutionalCode(generator_matrix=[[0o117, 0o155]])
    >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
    (2, 1, 6)

    >>> code = komm.ConvolutionalCode(generator_matrix=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]])
    >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
    (3, 2, 7)

    >>> code = komm.ConvolutionalCode(generator_matrix=[[0o7, 0o5]])
    >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
    (2, 1, 2)
    >>> code._outgoing_states
    {0: [0, 1], 1: [2, 3], 2: [0, 1], 3: [2, 3]}
    >>> code._outgoing_outputs
    {0: [array([0, 0]), array([1, 1])],
     1: [array([1, 0]), array([0, 1])],
     2: [array([1, 1]), array([0, 0])],
     3: [array([0, 1]), array([1, 0])]}
    >>> code._input_edge
    {(0, 0): array([0]),
     (0, 1): array([1]),
     (1, 2): array([0]),
     (1, 3): array([1]),
     (2, 0): array([0]),
     (2, 1): array([1]),
     (3, 2): array([0]),
     (3, 3): array([1])}
    """

    def __init__(self, generator_matrix):
        """ Constructor for the class. It expects the following parameter:

        :code:`generator_matrix` : 2D-array of :obj:`int`
            Generator matrix :math:`G(D)` in polynomial form, which is a :math:`k \\times n` matrix with integer entries representing binary polynomials (:obj:`BinaryPolynomial`).
        """
        self._generator_matrix = np.empty_like(generator_matrix, dtype=np.object)
        for i, row in enumerate(generator_matrix):
            self._generator_matrix[i] = [BinaryPolynomial(x) for x in row]

        self._num_input_bits, self._num_output_bits = self._generator_matrix.shape
        self._constraint_lengths = np.max(np.vectorize(lambda x: x.degree)(self._generator_matrix), axis=1)
        self._overall_constraint_length = np.sum(self._constraint_lengths)
        self._num_states = 2**self._overall_constraint_length

        self._init_finite_state_machine()

    def __repr__(self):
        args = str(np.vectorize(oct)(self._generator_matrix).tolist()).replace("'", "")
        return '{}({})'.format(self.__class__.__name__, args)

    def _init_finite_state_machine(self):
        k, n = self._num_input_bits, self._num_output_bits
        nu, num_states = self._overall_constraint_length, self._num_states

        bits = np.empty(k + nu, dtype=np.int)
        taps = np.empty((n, k + nu), dtype=np.int)

        i_indices = np.concatenate(([0], np.cumsum(self._constraint_lengths + 1)[:-1]))
        s0_indices = np.setdiff1d(np.arange(k + nu), i_indices)
        s1_indices = s0_indices - 1

        for j in range(n):
            taps[j, :] = np.concatenate([self._generator_matrix[i, j].coefficients(width=self._constraint_lengths[i] + 1)
                                         for i in range(k)])

        self._outgoing_states = {s: [] for s in range(num_states)}
        self._outgoing_outputs = {s: [] for s in range(num_states)}
        self._input_edge = {}

        i_indices = np.concatenate(([0], np.cumsum(self._constraint_lengths + 1)[:-1]))
        s0_indices = np.setdiff1d(np.arange(k + nu), i_indices)
        s1_indices = s0_indices - 1

        for s0, s0_bin in enumerate(binary_iterator(nu)):
            for i, i_bin in enumerate(binary_iterator(k)):
                bits[i_indices] = i_bin
                bits[s0_indices] = s0_bin
                s1_bin = bits[s1_indices]
                s1 = binlist2int(s1_bin)
                o_bin = np.dot(bits, taps.T) % 2
                self._outgoing_states[s0].append(s1)
                self._outgoing_outputs[s0].append(o_bin)
                self._input_edge[s0, s1] = i_bin

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

            \\nu_i = \\max_{0 \\leq j < n} \\{ \\deg g_{ij}(D) \\},

        where :math:`g_{ij}(D)` denotes the element in row :math:`i` and column :math:`j` of :math:`G(D)`, for :math:`i \\in [0 : k)` and :math:`j \\in [0 : n)`. This property is read-only.
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
    def num_states(self):
        """
        The number of states of the finite-state machine. It is given by :math:`2^{\\nu}`, where :math:`\\nu` is the overall constraint length of the code. This property is read-only.
        """
        return self._num_states

    @property
    def memory_order(self):
        """
        The memory order :math:`m` of the code. It is given by

        .. math::

            m = \\max_{0 \\leq i < k} \\nu_i

        This property is read-only.
        """
        return  np.max(self._constraint_lengths)

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
        k, n = self._num_input_bits, self._num_output_bits
        frame_size = message.size // k
        codeword = np.empty(n * frame_size, dtype=np.int)
        state = initial_state
        for (t, m) in enumerate(np.reshape(message, newshape=(frame_size, k))):
            m_int = binlist2int(m)
            codeword[n*t : n*(t+1)] = self._outgoing_outputs[state][m_int]
            state = self._outgoing_states[state][m_int]
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
        hamming_dist = lambda o, r: np.count_nonzero(o != r)
        return self._viterbi(recvword, inf=recvword.size, dist_fun=hamming_dist)

    @tag(name='Viterbi (soft)', input_type='soft', target='message')
    def _decode_viterbi_soft(self, recvword):
        return self._viterbi(recvword, inf=np.inf, dist_fun=np.dot)

    def _viterbi(self, recvword, inf, dist_fun, initial_state=0):   #&* final-state (None or int)
        k, n = self._num_input_bits, self._num_output_bits
        num_states = self._num_states
        frame_size = recvword.size // n

        choices = np.empty((num_states, frame_size), dtype=np.int)
        metrics = np.full((num_states, frame_size + 1), fill_value=inf, dtype=type(inf))
        metrics[initial_state, 0] = 0
        for (t, r) in enumerate(np.reshape(recvword, newshape=(frame_size, n))):
            for s0 in range(num_states):
                for i, (s1, o) in enumerate(zip(self._outgoing_states[s0], self._outgoing_outputs[s0])):
                    candidate_metrics = metrics[s0, t] + dist_fun(o, r)
                    if candidate_metrics < metrics[s1, t+1]:
                        metrics[s1, t+1] = candidate_metrics
                        choices[s1, t] = s0

        # Backtrack
        s1 = 0  #@$% final_state
        message_hat = np.empty(k * frame_size, dtype=np.int)
        for t in range(frame_size - 1, -1, -1):
            s0 = choices[s1, t]
            message_hat[k*t : k*(t+1)] = self._input_edge[s0, s1]
            s1 = s0

        return message_hat

    def _decode_bcjr():
        pass

    def _default_decoder(self, dtype):
        if dtype == np.int:
            return 'viterbi_hard'
        elif dtype == np.float:
            return 'viterbi_soft'
