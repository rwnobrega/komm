import numpy as np

from .._algebra import BinaryPolynomial, BinaryPolynomialFraction
from .._finite_state_machine import FiniteStateMachine
from .._util import binlist2int, int2binlist


class ConvolutionalCode:
    r"""
    Binary convolutional code. It is characterized by a *matrix of feedforward polynomials* $P(D)$, of shape $k \times n$, and (optionally) by a *vector of feedback polynomials* $q(D)$, of length $k$. The element in row $i$ and column $j$ of $P(D)$ is denoted by $p_{i,j}(D)$, and the element in position $i$ of $q(D)$ is denoted by $q_i(D)$; they are binary polynomials (:class:`BinaryPolynomial`) in $D$. The parameters $k$ and $n$ are the number of input and output bits per block, respectively.

    The *transfer function matrix* (also known as *transform-domain generator matrix*) $G(D)$ of the convolutional code, of shape $k \times n$, is such that the element in row $i$ and column $j$ is given by

    .. math::
       g_{i,j}(D) = \frac{p_{i,j}(D)}{q_{i}(D)},

    for $i \in [0 : k)$ and $j \in [0 : n)$.

    .. rubric:: Constraint lengths and related parameters

    The *constraint lengths* of the code are defined by

    .. math::
       \nu_i = \max \\{ \deg p_{i,0}(D), \deg p_{i,1}(D), \ldots, \deg p_{i,n-1}(D), \deg q_i(D) \\},

    for $i \in [0 : k)$.

    The *overall constraint length* of the code is defined by

    .. math::
       \nu = \sum_{0 \leq i < k} \nu_i.

    The *memory order* of the code is defined by

    .. math::
       \mu = \max_{0 \leq i < k} \nu_i.

    .. rubric:: Space-state representation

    A convolutional code may also be described via the *space-state representation*. Let $\mathbf{u}_t = (u_t^{(0)}, u_t^{(1)}, \ldots, u_t^{(k-1)})$ be the input block, $\mathbf{v}_t = (v_t^{(0)}, v_t^{(1)}, \ldots, v_t^{(n-1)})$ be the output block, and $\mathbf{s}_t = (s_t^{(0)}, s_t^{(1)}, \ldots, s_t^{(\nu-1)})$ be the state, all defined at time instant $t$. Then,

    .. math::
       \begin{aligned}
          \mathbf{s}_{t+1} & = \mathbf{s}_t A + \mathbf{u}_t B, \\
          \mathbf{v}_{t} & = \mathbf{s}_t C + \mathbf{u}_t D,
       \end{aligned}

    where $A$ is the $\nu \times \nu$ *state matrix*, $B$ is the $k \times \nu$ *control matrix*, $C$ is the $\nu \times n$ *observation matrix*, and $D$ is the $k \times n$ *transition matrix*.

    .. rubric:: Table of convolutional codes

    The table below lists optimal convolutional codes with parameters $(n,k) = (2,1)$ and $(n,k) = (3,1)$, for small values of the overall constraint length $\nu$. For more details, see :cite:`Lin.Costello.04` (Sec. 12.3).

    =================================  ======================================
     Parameters $(n, k, \nu)$    Transfer function matrix $G(D)$
    =================================  ======================================
     $(2, 1, 1)$                  :code:`[[0o1, 0o3]]`
     $(2, 1, 2)$                  :code:`[[0o5, 0o7]]`
     $(2, 1, 3)$                  :code:`[[0o13, 0o17]]`
     $(2, 1, 4)$                  :code:`[[0o27, 0o31]]`
     $(2, 1, 5)$                  :code:`[[0o53, 0o75]]`
     $(2, 1, 6)$                  :code:`[[0o117, 0o155]]`
     $(2, 1, 7)$                  :code:`[[0o247, 0o371]]`
     $(2, 1, 8)$                  :code:`[[0o561, 0o753]]`
    =================================  ======================================

    =================================  ======================================
     Parameters $(n, k, \nu)$    Transfer function matrix $G(D)$
    =================================  ======================================
     $(3, 1, 1)$                  :code:`[[0o1, 0o3, 0o3]]`
     $(3, 1, 2)$                  :code:`[[0o5, 0o7, 0o7]]`
     $(3, 1, 3)$                  :code:`[[0o13, 0o15, 0o17]]`
     $(3, 1, 4)$                  :code:`[[0o25, 0o33, 0o37]]`
     $(3, 1, 5)$                  :code:`[[0o47, 0o53, 0o75]]`
     $(3, 1, 6)$                  :code:`[[0o117, 0o127, 0o155]]`
     $(3, 1, 7)$                  :code:`[[0o255, 0o331, 0o367]]`
     $(3, 1, 8)$                  :code:`[[0o575, 0o623, 0o727]]`
    =================================  ======================================

    References:

        1. :cite:`Johannesson.Zigangirov.15`
        2. :cite:`Lin.Costello.04`
        3. :cite:`Weiss.01`
    """

    def __init__(self, feedforward_polynomials, feedback_polynomials=None):
        r"""
        Constructor for the class.

        Parameters:

            feedforward_polynomials (2D-array of (:obj:`BinaryPolynomial` or :obj:`int`)): The matrix of feedforward polynomials $P(D)$, which is a $k \times n$ matrix whose entries are either binary polynomials (:obj:`BinaryPolynomial`) or integers to be converted to the former.

            feedback_polynomials (1D-array of  (:obj:`BinaryPolynomial` or :obj:`int`), optional): The vector of feedback polynomials $q(D)$, which is a $k$-vector whose entries are either binary polynomials (:obj:`BinaryPolynomial`) or integers to be converted to the former. The default value corresponds to no feedback, that is, $q_i(D) = 1$ for all $i \in [0 : k)$.

        Examples:

            The convolutional code with encoder depicted in the figure below has parameters $(n, k, \nu) = (2, 1, 6)$; its transfer function matrix is given by

            .. math::
               G(D) =
               \begin{bmatrix}
                   D^6 + D^3 + D^2 + D + 1  &  D^6 + D^5 + D^3 + D^2 + 1
               \end{bmatrix},

            yielding :code:`feedforward_polynomials = [[0b1001111, 0b1101101]] = [[0o117, 0o155]] = [[79, 109]]`.

            .. image:: figures/cc_2_1_6.svg
               :alt: Convolutional encoder example.
               :align: center

            >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o117, 0o155]])
            >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
            (2, 1, 6)

            The convolutional code with encoder depicted in the figure below has parameters $(n, k, \nu) = (3, 2, 7)$; its transfer function matrix is given by

            .. math::
               G(D) =
               \begin{bmatrix}
                   D^4 + D^3 + 1  &  D^4 + D^2 + D + 1  &  0 \\
                   0  &  D^3 + D  &  D^3 + D^2 + 1 \\
               \end{bmatrix},

            yielding :code:`feedforward_polynomials = [[0b11001, 0b10111, 0b00000], [0b0000, 0b1010, 0b1101]] = [[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]] = [[25, 23, 0], [0, 10, 13]]`.

            .. image:: figures/cc_3_2_7.svg
               :alt: Convolutional encoder example.
               :align: center

            >>> code = komm.ConvolutionalCode(feedforward_polynomials=[[0o31, 0o27, 0o00], [0o00, 0o12, 0o15]])
            >>> (code.num_output_bits, code.num_input_bits, code.overall_constraint_length)
            (3, 2, 7)

            The convolutional code with feedback encoder depicted in the figure below has parameters $(n, k, \nu) = (2, 1, 4)$; its transfer function matrix is given by

            .. math::
               G(D) =
               \begin{bmatrix}
                   1  &  \dfrac{D^4 + D^3 + 1}{D^4 + D^2 + D + 1}
               \end{bmatrix},

            yielding :code:`feedforward_polynomials = [[0b10111, 0b11001]] = [[0o27, 0o31]] = [[23, 25]]` and :code:`feedback_polynomials = [0o27]`.

            .. image:: figures/cc_2_1_4_fb.svg
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
            self._feedback_polynomials = np.array([BinaryPolynomial(0b1) for _ in range(k)], dtype=object)
            self._constructed_from = "no_feedback_polynomials"
        else:
            self._feedback_polynomials = np.empty_like(feedback_polynomials, dtype=object)
            for i, q in np.ndenumerate(feedback_polynomials):
                self._feedback_polynomials[i] = BinaryPolynomial(q)
            self._constructed_from = "feedback_polynomials"

        nus = np.empty(k, dtype=int)
        for i, (ps, q) in enumerate(zip(self._feedforward_polynomials, self._feedback_polynomials)):
            nus[i] = max(np.amax([p.degree for p in ps]), q.degree)

        self._num_input_bits = k
        self._num_output_bits = n
        self._constraint_lengths = nus
        self._overall_constraint_length = np.sum(nus)
        self._memory_order = np.amax(nus)

        self._transfer_function_matrix = np.empty((k, n), dtype=object)
        for (i, j), p in np.ndenumerate(feedforward_polynomials):
            q = self._feedback_polynomials[i]
            self._transfer_function_matrix[i, j] = BinaryPolynomialFraction(p) / BinaryPolynomialFraction(q)

        self._setup_finite_state_machine_direct_form()
        self._setup_space_state_representation()

    def __repr__(self):
        feedforward_polynomials_str = str(np.vectorize(str)(self._feedforward_polynomials).tolist()).replace("'", "")
        args = "feedforward_polynomials={}".format(feedforward_polynomials_str)
        if self._constructed_from == "feedback_polynomials":
            feedback_polynomials_str = str(np.vectorize(str)(self._feedback_polynomials).tolist()).replace("'", "")
            args = "{}, feedback_polynomials={}".format(args, feedback_polynomials_str)
        return "{}({})".format(self.__class__.__name__, args)

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

        bits = np.empty(k + nu, dtype=int)
        next_states = np.empty((2**nu, 2**k), dtype=int)
        outputs = np.empty((2**nu, 2**k), dtype=int)

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

        self._state_matrix = np.empty((nu, nu), dtype=int)
        self._observation_matrix = np.empty((nu, n), dtype=int)
        for i in range(nu):
            s0 = 2**i
            s1 = self._finite_state_machine.next_states[s0, 0]
            y = self._finite_state_machine.outputs[s0, 0]
            self._state_matrix[i, :] = int2binlist(s1, width=nu)
            self._observation_matrix[i, :] = int2binlist(y, width=n)

        self._control_matrix = np.empty((k, nu), dtype=int)
        self._transition_matrix = np.empty((k, n), dtype=int)
        for i in range(k):
            x = 2**i
            s1 = self._finite_state_machine.next_states[0, x]
            y = self._finite_state_machine.outputs[0, x]
            self._control_matrix[i, :] = int2binlist(s1, width=nu)
            self._transition_matrix[i, :] = int2binlist(y, width=n)

    @property
    def num_input_bits(self):
        r"""
        The number of input bits per block, $k$. This property is read-only.
        """
        return self._num_input_bits

    @property
    def num_output_bits(self):
        r"""
        The number of output bits per block, $n$. This property is read-only.
        """
        return self._num_output_bits

    @property
    def constraint_lengths(self):
        r"""
        The constraint lengths $\nu_i$ of the code, for $i \in [0 : k)$. This is a 1D-array of :obj:`int`. This property is read-only.
        """
        return self._constraint_lengths

    @property
    def overall_constraint_length(self):
        r"""
        The overall constraint length $\nu$ of the code. This property is read-only.
        """
        return self._overall_constraint_length

    @property
    def memory_order(self):
        r"""
        The memory order $\mu$ of the code. This property is read-only.
        """
        return self._memory_order

    @property
    def feedforward_polynomials(self):
        r"""
        The matrix of feedforward polynomials $P(D)$ of the code. This is a $k \times n$ array of :obj:`BinaryPolynomial`. This property is read-only.
        """
        return self._feedforward_polynomials

    @property
    def feedback_polynomials(self):
        r"""
        The vector of feedback polynomials $q(D)$ of the code. This is a $k$-array of :obj:`BinaryPolynomial`. This property is read-only.
        """
        return self._feedback_polynomials

    @property
    def transfer_function_matrix(self):
        r"""
        The transfer function matrix $G(D)$ of the code. This is a $k \times n$ array of :obj:`BinaryPolynomialFraction`. This property is read-only.
        """
        return self._transfer_function_matrix

    @property
    def finite_state_machine(self):
        r"""
        The finite-state machine of the code.
        """
        return self._finite_state_machine

    @property
    def state_matrix(self):
        r"""
        The state matrix $A$ of the state-space representation. This is a $\nu \times \nu$ array of integers in $\\{ 0, 1 \\}$. This property is read-only.
        """
        return self._state_matrix

    @property
    def control_matrix(self):
        r"""
        The control matrix $B$ of the state-space representation. This is a $k \times \nu$ array of integers in $\\{ 0, 1 \\}$. This property is read-only.
        """
        return self._control_matrix

    @property
    def observation_matrix(self):
        r"""
        The observation matrix $C$ of the state-space representation. This is a $\nu \times n$ array of integers in $\\{ 0, 1 \\}$. This property is read-only.
        """
        return self._observation_matrix

    @property
    def transition_matrix(self):
        r"""
        The transition matrix $D$ of the state-space representation. This is a $k \times n$ array of integers in $\\{ 0, 1 \\}$. This property is read-only.
        """
        return self._transition_matrix
