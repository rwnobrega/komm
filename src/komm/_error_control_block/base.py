from abc import ABC, abstractmethod
from functools import cache, cached_property

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.bit_operations import bits_to_int
from .._util.decorators import blockwise
from .._util.matrices import boolean_matmul, matmul, pseudo_inverse, row_span
from ..types import Array2D
from .util import macwilliams_transform, span_weight_distribution


class BlockCode(ABC):
    @cached_property
    @abstractmethod
    def length(self) -> int:
        r"""
        The length $n$ of the code.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def dimension(self) -> int:
        r"""
        The dimension $k$ of the code.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def redundancy(self) -> int:
        r"""
        The redundancy $m$ of the code.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def rate(self) -> float:
        r"""
        The rate $R = k/n$ of the code.
        """
        return self.dimension / self.length

    @cached_property
    @abstractmethod
    def generator_matrix(self) -> Array2D[np.integer]:
        r"""
        The generator matrix $G \in \mathbb{B}^{k \times n}$ of the code.
        """
        raise NotImplementedError

    @cached_property
    def generator_matrix_right_inverse(self) -> Array2D[np.integer]:
        return pseudo_inverse(self.generator_matrix)

    @cached_property
    @abstractmethod
    def check_matrix(self) -> Array2D[np.integer]:
        r"""
        The check matrix $H \in \mathbb{B}^{m \times n}$ of the code.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Applies the encoding mapping $\Enc : \mathbb{B}^k \to \mathbb{B}^n$ of the code. This method takes one or more sequences of messages and returns their corresponding codeword sequences.

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $k$, or a multidimensional array where the last dimension is a multiple of $k$.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension expanded from $bk$ to $bn$, where $b$ is a positive integer.
        """

        @blockwise(self.dimension)
        def encode(u: npt.NDArray[np.integer]):
            v = matmul(u, self.generator_matrix)
            return v

        return encode(input)

    def project_word(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.length)
        def project(v: npt.NDArray[np.integer]):
            u = matmul(v, self.generator_matrix_right_inverse)
            return u

        return project(input)

    def project_word_with_erasures(
        self, input: npt.ArrayLike
    ) -> npt.NDArray[np.integer]:
        # Conservative: may erase recoverable message bits.
        @blockwise(self.length)
        def project(v: npt.NDArray[np.integer]):
            G_r_inv = self.generator_matrix_right_inverse
            erased = v == 2
            u = matmul(np.where(erased, 0, v), G_r_inv)
            u[boolean_matmul(erased, G_r_inv)] = 2
            return u

        return project(input)

    @abstractmethod
    def inverse_encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Applies the inverse encoding partial mapping $\Enc^{-1} : \mathbb{B}^n \rightharpoonup \mathbb{B}^k$ of the code. This method takes one or more sequences of codewords and returns their corresponding message sequences.

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $n$, or a multidimensional array where the last dimension is a multiple of $n$.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension contracted from $bn$ to $bk$, where $b$ is a positive integer.

        Raises:
            ValueError: If the input contains any invalid codewords.
        """
        s = self.check(input)
        if not np.all(s == 0):
            raise ValueError("one or more inputs in 'v' are not valid codewords")
        return self.project_word(input)

    @abstractmethod
    def check(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Applies the check mapping $\mathrm{Chk} : \mathbb{B}^n \to \mathbb{B}^m$ of the code. This method takes one or more sequences of received words and returns their corresponding syndrome sequences.

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $n$, or a multidimensional array where the last dimension is a multiple of $n$.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension contracted from $bn$ to $bm$, where $b$ is a positive integer.
        """

        @blockwise(self.length)
        def check(r: npt.NDArray[np.integer]):
            s = matmul(r, self.check_matrix.T)
            return s

        return check(input)

    @cache
    @abstractmethod
    def codewords(self) -> Array2D[np.integer]:
        r"""
        Returns the codewords of the code. This is a $2^k \times n$ matrix whose rows are all the codewords. The codeword in row $i$ corresponds to the message obtained by expressing $i$ in binary with $k$ bits (LSB-first).
        """
        n = self.length
        G = self.generator_matrix
        # Batches of up to 2**10 codewords.
        low, high = row_span(G[:10]), row_span(G[10:])
        codewords = np.empty((len(high), len(low), n), dtype=int)
        pbar = tqdm(high, desc="Generating codewords", delay=2.5, unit_scale=len(low))
        for i, offset in enumerate(pbar):
            codewords[i] = low ^ offset
        self._cached_codewords = True
        return codewords.reshape(-1, n)

    @cache
    @abstractmethod
    def codeword_weight_distribution(self) -> list[int]:
        r"""
        Returns the codeword weight distribution of the code. This is the list $A_0, A_1, \ldots, A_n$, in which $A_w$ is the number of codewords of Hamming weight $w$.

        If $m < k$, this method enumerates the $2^m$ codewords of the dual code, spanned by the rows of the check matrix $H$, and applies the MacWilliams identity
        $$
            \sum_{w=0}^n A_w z^w = \frac{1}{2^m} \sum_{w=0}^n B_w (1 - z)^w (1 + z)^{n - w},
        $$
        where $B_w$ is the number of codewords of Hamming weight $w$ in the dual code. Otherwise, it enumerates the $2^k$ codewords of the code. For more details, see <cite>LC04, Sec. 3.6</cite>.
        """
        n = self.length
        if hasattr(self, "_cached_codewords"):
            weights = np.sum(self.codewords(), axis=1)
            return np.bincount(weights, minlength=n + 1).tolist()
        if self.redundancy < self.dimension:
            # Fewer dual codewords: MacWilliams identity.
            dual = span_weight_distribution(self.check_matrix)
            return macwilliams_transform(dual)
        return span_weight_distribution(self.generator_matrix)

    @cache
    @abstractmethod
    def minimum_distance(self) -> int:
        r"""
        Returns the minimum distance $d$ of the code. This is equal to the minimum Hamming weight of the non-zero codewords.
        """
        return int(np.flatnonzero(self.codeword_weight_distribution())[1])

    @cache
    @abstractmethod
    def coset_leaders(self) -> Array2D[np.integer]:
        r"""
        Returns the coset leaders of the code. This is a $2^m \times n$ matrix whose rows are all the coset leaders. The coset leader in row $i$ corresponds to the syndrome obtained by expressing $i$ in binary with $m$ bits (LSB-first), and whose Hamming weight is minimal. This may be used as a LUT for syndrome-based decoding.

        Of all minimal-weight words in each coset, this method returns the one whose support comes first in lexicographic order.
        """
        m, n = self.redundancy, self.length
        H_cols = bits_to_int(self.check_matrix.T.ravel(), width=m)
        visited = np.zeros(2**m, dtype=bool)
        visited[0] = True
        syndromes = np.array([0])
        leaders = np.zeros((2**m, n), dtype=int)
        pbar = tqdm(total=2**m, desc="Generating coset leaders", delay=2.5, initial=1)
        while not visited.all():
            next_syndromes: list[npt.NDArray[np.integer]] = []
            for j, h in enumerate(H_cols):
                # XOR by h is injective: no repeats.
                candidates = syndromes ^ h
                candidates = candidates[~visited[candidates]]
                visited[candidates] = True
                # Parent leader, plus bit j.
                leaders[candidates] = leaders[candidates ^ h]
                leaders[candidates, j] = 1
                next_syndromes.append(candidates)
                pbar.update(candidates.size)
            syndromes = np.concatenate(next_syndromes)
        pbar.close()
        self._cached_coset_leaders = True
        return leaders

    @cache
    @abstractmethod
    def coset_leader_weight_distribution(self) -> list[int]:
        r"""
        Returns the coset leader weight distribution of the code. This is the list $\alpha_0, \alpha_1, \ldots, \alpha_n$, in which $\alpha_w$ is the number of coset leaders of Hamming weight $w$.
        """
        m, n = self.redundancy, self.length
        if hasattr(self, "_cached_coset_leaders"):
            weights = np.sum(self.coset_leaders(), axis=1)
            return np.bincount(weights, minlength=n + 1).tolist()
        H_cols = bits_to_int(self.check_matrix.T.ravel(), width=m)
        visited = np.zeros(2**m, dtype=bool)
        visited[0] = True
        syndromes = np.array([0])
        weight = 0
        distribution = np.zeros(n + 1, dtype=int)
        distribution[0] = 1
        desc = "Computing coset leader weight distribution"
        pbar = tqdm(total=2**m, desc=desc, delay=2.5, initial=1)
        while not visited.all():
            next_syndromes: list[npt.NDArray[np.integer]] = []
            for h in H_cols:
                # XOR by h is injective: no repeats.
                candidates = syndromes ^ h
                candidates = candidates[~visited[candidates]]
                visited[candidates] = True
                next_syndromes.append(candidates)
                pbar.update(candidates.size)
            syndromes = np.concatenate(next_syndromes)
            weight += 1
            distribution[weight] = syndromes.size
        pbar.close()
        return distribution.tolist()

    @cache
    @abstractmethod
    def packing_radius(self) -> int:
        r"""
        Returns the packing radius of the code. This is also called the *error-correcting capability* of the code, and is equal to $\lfloor (d - 1) / 2 \rfloor$.
        """
        return (self.minimum_distance() - 1) // 2

    @cache
    @abstractmethod
    def covering_radius(self) -> int:
        r"""
        Returns the covering radius of the code. This is equal to the maximum Hamming weight of the coset leaders.
        """
        return int(np.flatnonzero(self.coset_leader_weight_distribution())[-1])
