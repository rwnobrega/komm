import numpy as np
import numpy.typing as npt

from .._util.decorators import blockwise, vectorize
from .._util.docs import mkdocstrings
from .BlockCode import BlockCode


@mkdocstrings(filters=["!.*"])
class PolarCode(BlockCode):
    r"""
    Polar (Arıkan) code. Let $\mu \geq 1$ be an integer, and $\mathcal{F}$ (called the _frozen bit indices_) be a subset of $[0 : 2^\mu)$. Define $\mathcal{A} = [0 : 2^\mu) \setminus \mathcal{F}$ (called the _active bit indices_). The polar code with parameters $(\mu, \mathcal{F})$ is the [linear block code](/ref/BlockCode) whose generator matrix is obtained by selecting the rows of the order-$2^\mu$ Walsh-Hadamard matrix,
    $$ H_{2^\mu} = \begin{bmatrix} 1 & 0 \\\\ 1 & 1 \end{bmatrix} ^ {\otimes \mu}, $$
    corresponding to the active bit indices, where $\otimes$ denotes the Kronecker product. The resulting code has the following parameters:

    - Length: $n = 2^{\mu}$
    - Dimension: $k = |\mathcal{A}|$
    - Redundancy: $m = |\mathcal{F}|$

    Parameters:
        mu: The parameter $\mu$ of the code.
        frozen: The frozen bit indices $\mathcal{F}$ of the code.

    Examples:
        >>> code = komm.PolarCode(4, [0, 1, 2, 3, 4, 8])
        >>> (code.length, code.dimension, code.redundancy)
        (16, 10, 6)
        >>> code.generator_matrix
        array([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
               [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
               [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        >>> code.minimum_distance()
        4
    """

    mu: int
    frozen: npt.NDArray[np.integer]

    def __init__(self, mu: int, frozen: npt.ArrayLike):
        self.mu = mu
        if not mu >= 1:
            raise ValueError("'mu' must be greater than or equal to 1")
        self.frozen = np.sort(frozen).astype(int)
        if not np.all((0 <= self.frozen) & (self.frozen < 2**mu)):
            raise ValueError("frozen bits must be between 0 and 2^mu - 1")
        if self.frozen.size != np.unique(self.frozen).size:
            raise ValueError("frozen bits must be unique")
        self.active = np.setdiff1d(np.arange(2**mu, dtype=int), self.frozen)
        hadamard = np.array([[1]])
        for _ in range(mu):
            hadamard = np.kron(hadamard, [[1, 0], [1, 1]]).astype(int)
        super().__init__(generator_matrix=hadamard[self.active])

    def __repr__(self) -> str:
        args = f"mu={self.mu}, frozen={self.frozen.tolist()}"
        return f"{self.__class__.__name__}({args})"

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.dimension)
        @vectorize
        def encode(u: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
            n = 2**self.mu
            v = np.zeros((n,), dtype=int)
            v[self.active] = u
            for i in range(self.mu):
                m = 2**i
                for i in range(0, n, 2 * m):
                    v[i : i + m] ^= v[i + m : i + 2 * m]
            return v

        return encode(input)
