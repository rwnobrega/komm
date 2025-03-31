from abc import ABC, abstractmethod
from functools import cache, cached_property
from itertools import combinations

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._util.bit_operations import bits_to_int, int_to_bits
from .._util.decorators import blockwise


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
    def generator_matrix(self) -> npt.NDArray[np.integer]:
        r"""
        The generator matrix $G \in \mathbb{B}^{k \times n}$ of the code.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def generator_matrix_right_inverse(self) -> npt.NDArray[np.integer]:
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def check_matrix(self) -> npt.NDArray[np.integer]:
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
            v = u @ self.generator_matrix % 2
            return v

        return encode(input)

    @abstractmethod
    def project_word(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        @blockwise(self.length)
        def project(v: npt.NDArray[np.integer]):
            u = v @ self.generator_matrix_right_inverse % 2
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
        Applies the check mapping $\mathrm{Chk}: \mathbb{B}^n \to \mathbb{B}^m$ of the code. This method takes one or more sequences of received words and returns their corresponding syndrome sequences.

        Parameters:
            input: The input sequence(s). Can be either a single sequence whose length is a multiple of $n$, or a multidimensional array where the last dimension is a multiple of $n$.

        Returns:
            output: The output sequence(s). Has the same shape as the input, with the last dimension contracted from $bn$ to $bm$, where $b$ is a positive integer.
        """

        @blockwise(self.length)
        def check(r: npt.NDArray[np.integer]):
            s = r @ self.check_matrix.T % 2
            return s

        return check(input)

    @cache
    @abstractmethod
    def codewords(self) -> npt.NDArray[np.integer]:
        r"""
        Returns the codewords of the code. This is a $2^k \times n$ matrix whose rows are all the codewords. The codeword in row $i$ corresponds to the message obtained by expressing $i$ in binary with $k$ bits (MSB in the right).
        """
        batch_size = 1024
        k, n = self.dimension, self.length
        codewords = np.empty((2**k, n), dtype=int)
        for i in tqdm(
            range(0, 2**k, batch_size), desc="Generating codewords", delay=2.5
        ):
            batch_end = min(i + batch_size, 2**k)
            js = np.arange(i, batch_end, dtype=np.uint64).reshape(-1, 1).view(np.uint8)
            messages_batch = np.unpackbits(js, axis=1, count=k, bitorder="little")
            codewords[i:batch_end] = self.encode(messages_batch)
        return codewords

    @cache
    @abstractmethod
    def codeword_weight_distribution(self) -> npt.NDArray[np.integer]:
        r"""
        Returns the codeword weight distribution of the code. This is an array of shape $(n + 1)$ in which element in position $w$ is equal to the number of codewords of Hamming weight $w$, for $w \in [0 : n]$.
        """
        return np.bincount(np.sum(self.codewords(), axis=1), minlength=self.length + 1)

    @cache
    @abstractmethod
    def minimum_distance(self) -> int:
        r"""
        Returns the minimum distance $d$ of the code. This is equal to the minimum Hamming weight of the non-zero codewords.
        """
        return int(np.flatnonzero(self.codeword_weight_distribution())[1])

    @cache
    @abstractmethod
    def coset_leaders(self) -> npt.NDArray[np.integer]:
        r"""
        Returns the coset leaders of the code. This is a $2^m \times n$ matrix whose rows are all the coset leaders. The coset leader in row $i$ corresponds to the syndrome obtained by expressing $i$ in binary with $m$ bits (MSB in the right), and whose Hamming weight is minimal. This may be used as a LUT for syndrome-based decoding.
        """
        m, n = self.redundancy, self.length
        leaders = np.full(2**m, -1)
        taken = 0
        with tqdm(total=2**m, desc="Generating coset leaders", delay=2.5) as pbar:
            for w in range(n + 1):
                for idx in combinations(range(n), w):
                    leader = np.zeros(n, dtype=int)
                    leader[list(idx)] = 1
                    syndrome = self.check(leader)
                    i = bits_to_int(syndrome)
                    if leaders[i] != -1:
                        continue
                    taken += 1
                    pbar.update()
                    leaders[i] = bits_to_int(leader)
                    if taken == 2**m:
                        return int_to_bits(leaders, n)

    @cache
    @abstractmethod
    def coset_leader_weight_distribution(self) -> npt.NDArray[np.integer]:
        r"""
        Returns the coset leader weight distribution of the code. This is an array of shape $(n + 1)$ in which element in position $w$ is equal to the number of coset leaders of weight $w$, for $w \in [0 : n]$.
        """
        return np.bincount(
            np.sum(self.coset_leaders(), axis=1), minlength=self.length + 1
        )

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
