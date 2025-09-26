from collections import defaultdict
from dataclasses import dataclass
from math import ceil, log

import numpy as np
import numpy.typing as npt

from .util import integer_to_symbols, symbols_to_integer


@dataclass
class LempelZiv77Code:
    r"""
    Lempel–Ziv 77 (LZ77 or LZ1) code. It is a lossless data compression algorithm that replaces repeated data with references to previous occurrences within a sliding window. The algorithm achieves compression by identifying matches between the current position and patterns within the search window, encoding them as triples `(distance, length, next_symbol)`. For more details, see <cite>Say06, Sec. 5.4.1</cite>.

    Parameters:
        source_cardinality: The source cardinality $S$. Must be an integer greater than or equal to $2$.
        target_cardinality: The target cardinality $T$. Must be an integer greater than or equal to $2$. Default is $2$ (binary).
        window_size: Sliding window size $W$. Must be an integer greater than or equal to $1$.
        lookahead_size: Lookahead buffer size $L$. Must be an integer greater than or equal to $1$.

    Encoding format (fixed-width per triple):
        d: distance in [0..W]  (0 means "no match")
        l: length   in [0..L]  (0 means "no match")
        c: next symbol in [0..S-1]

        Each field is emitted in base-T using:
            D = ceil(log(W+1, T)) symbols for d
            Lw = ceil(log(L+1, T)) symbols for l
            M = ceil(log(S,   T)) symbols for c

    Examples:
        >>> lz77 = komm.LempelZiv77Code(
        ...     source_cardinality=2,
        ...     target_cardinality=2,
        ...     window_size=16,
        ...     lookahead_size=4,
        ... )
    """

    source_cardinality: int
    target_cardinality: int
    window_size: int
    lookahead_size: int

    def __post_init__(self) -> None:
        if self.source_cardinality < 2:
            raise ValueError("'source_cardinality' must be at least 2")
        if self.target_cardinality < 2:
            raise ValueError("'target_cardinality' must be at least 2")
        if self.window_size < 1:
            raise ValueError("'window_size' must be at least 1")
        if self.lookahead_size < 1:
            raise ValueError("'lookahead_size' must be at least 1")

        # Precompute field widths in base T.
        T: int = self.target_cardinality
        S: int = self.source_cardinality
        self._D: int = max(1, ceil(float(log(self.window_size + 1, T))))
        self._Lw: int = max(1, ceil(float(log(self.lookahead_size + 1, T))))
        self._M: int = max(1, ceil(float(log(S, T))))

        # Initialize hash-based matcher
        self._matcher: _HashMatcher = _HashMatcher(
            min_match_length=3, max_hash_entries=10, pattern_length=3
        )

    def encode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Encodes a sequence of source symbols using the LZ77 encoding algorithm.

        Parameters:
            input: The sequence of symbols to be encoded. Must be a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code).

        Returns:
            output: The sequence of encoded symbols. It is a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code).

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     source_cardinality=2,
            ...     target_cardinality=2,
            ...     window_size=4,
            ...     lookahead_size=2,
            ... )
            >>> lz77.encode([0, 0, 0, 0, 0, 0, 0])
            array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        """
        triples = self.source_to_triples(input)
        output = self.triples_to_target(triples)
        return output

    def decode(self, input: npt.ArrayLike) -> npt.NDArray[np.integer]:
        r"""
        Decodes a sequence of encoded symbols using the LZ77 decoding algorithm.

        Parameters:
            input: The sequence of symbols to be decoded. Must be a 1D-array with elements in $[0:T)$ (where $T$ is the target cardinality of the code). Also, the sequence must be a valid output of the `encode` method.

        Returns:
            output: The sequence of decoded symbols. It is a 1D-array with elements in $[0:S)$ (where $S$ is the source cardinality of the code).

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     source_cardinality=2,
            ...     target_cardinality=2,
            ...     window_size=4,
            ...     lookahead_size=2,
            ... )
            >>> lz77.decode([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
            array([0, 0, 0, 0, 0, 0, 0])
        """
        triples = self.target_to_triples(input)
        output = self.triples_to_source(triples)
        return output

    def source_to_triples(self, input: npt.ArrayLike) -> list[tuple[int, int, int]]:
        r"""
        Convert source symbols to list of LZ77 triples (d, l, c) using hash optimization.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     source_cardinality=2,
            ...     target_cardinality=2,
            ...     window_size=4,
            ...     lookahead_size=2,
            ... )
            >>> lz77.source_to_triples([0, 0, 0, 0, 0, 0, 0])
            [(0, 0, 0), (1, 2, 0), (4, 2, 0)]
        """
        x: npt.NDArray[np.integer] = np.asarray(input, dtype=int)
        if x.ndim != 1:
            raise ValueError("'input' must be a 1D array")
        if x.size == 0:
            return []
        if np.any((x < 0) | (x >= self.source_cardinality)):
            raise ValueError("input symbols out of range")

        W: int = self.window_size
        L: int = self.lookahead_size
        triples: list[tuple[int, int, int]] = []
        n: int = x.size
        i: int = 0

        # Initialize the hash matcher with current data
        self._matcher.initialize(x)

        while i < n:
            if i == n - 1:
                triples.append((0, 0, int(x[i])))
                i += 1
                continue

            win_start: int = max(0, i - W)
            window: npt.NDArray[np.integer] = x[win_start:i]
            max_l: int = min(L, n - i - 1)
            lookahead: npt.NDArray[np.integer] = x[i : i + max_l]

            # Update hash table with current position so future matches can reference it
            self._matcher.update_hash_table(i)

            # Find longest match using hash optimization
            d: int
            l: int
            d, l = self._matcher.find_longest_match_optimized(window, lookahead, i)

            if d == 0 or l == 0:
                d, l, c = 0, 0, int(x[i])
                step: int = 1
            else:
                c: int = int(x[i + l])

                # Feed the hash table with every position covered by the match so
                # future iterations can discover longer references. Without this
                # bookkeeping, we would skip over matched symbols and never make
                # them available for upcoming searches, severely hurting both
                # compression ratio and speed.
                for offset in range(1, l + 1):
                    self._matcher.update_hash_table(i + offset)

                step = l + 1

            triples.append((d, l, c))
            i += step

        return triples

    def triples_to_target(
        self, triples: list[tuple[int, int, int]]
    ) -> npt.NDArray[np.integer]:
        r"""
        Convert list of triples to transmission symbol stream.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     source_cardinality=2,
            ...     target_cardinality=2,
            ...     window_size=4,
            ...     lookahead_size=2,
            ... )
            >>> lz77.triples_to_target([(0, 0, 0), (1, 2, 0), (4, 2, 0)])
            array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        """
        T: int = self.target_cardinality
        D: int = self._D
        Lw: int = self._Lw
        M: int = self._M
        out: list[int] = []

        for d, l, c in triples:
            out.extend(integer_to_symbols(d, base=T, width=D))
            out.extend(integer_to_symbols(l, base=T, width=Lw))
            out.extend(integer_to_symbols(c, base=T, width=M))

        return np.array(out, dtype=int)

    def target_to_triples(self, input: npt.ArrayLike) -> list[tuple[int, int, int]]:
        r"""
        Parse transmission stream back to list of triples.

        Examples:
            >>> lz77 = komm.LempelZiv77Code(
            ...     source_cardinality=2,
            ...     target_cardinality=2,
            ...     window_size=4,
            ...     lookahead_size=2,
            ... )
            >>> lz77.target_to_triples([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
            [(0, 0, 0), (1, 2, 0), (4, 2, 0)]
        """
        T: int = self.target_cardinality
        D: int = self._D
        Lw: int = self._Lw
        M: int = self._M
        y: npt.NDArray[np.integer] = np.asarray(input, dtype=int)

        if y.ndim != 1:
            raise ValueError("'input' must be a 1D array")
        if y.size == 0:
            return []
        if np.any((y < 0) | (y >= T)):
            raise ValueError("encoded symbols out of range for base T")

        triples: list[tuple[int, int, int]] = []
        i: int = 0
        triple_syms: int = D + Lw + M

        while i + triple_syms <= y.size:
            d: int = int(symbols_to_integer(y[i : i + D], base=T))
            i += D
            l: int = int(symbols_to_integer(y[i : i + Lw], base=T))
            i += Lw
            c: int = int(symbols_to_integer(y[i : i + M], base=T))
            i += M
            triples.append((d, l, c))

        if i != y.size:
            raise ValueError(
                "Invalid stream: leftover symbols not forming a complete triple"
            )

        return triples

    def triples_to_source(
        self, triples: list[tuple[int, int, int]]
    ) -> npt.NDArray[np.integer]:
        r"""
        Reconstruct original message from list of triples.

        >>> lz77 = komm.LempelZiv77Code(
        ...     source_cardinality=2,
        ...     target_cardinality=2,
        ...     window_size=4,
        ...     lookahead_size=2,
        ... )
        >>> lz77.triples_to_source([(0, 0, 0), (1, 2, 0), (4, 2, 0)])
        array([0, 0, 0, 0, 0, 0, 0])
        """
        S: int = self.source_cardinality
        out: list[int] = []

        for d, l, c in triples:
            if c >= S:
                raise ValueError(f"Invalid stream: symbol c={c} not in [0,{S - 1}]")
            if d == 0 and l != 0:
                raise ValueError("Invalid stream: d=0 must imply l=0")
            if l > self.lookahead_size:
                raise ValueError("Invalid stream: length exceeds lookahead_size")

            if d == 0 and l == 0:
                out.append(c)
                continue

            start: int = len(out) - d
            if start < 0:
                raise ValueError("Invalid stream: distance exceeds produced output")

            for k in range(l):
                out.append(out[start + k])

            out.append(c)

        return np.array(out, dtype=int)


class _HashMatcher:
    def __init__(
        self,
        min_match_length: int = 3,
        max_hash_entries: int = 10,
        pattern_length: int = 3,
    ) -> None:
        self.min_match_length: int = min_match_length
        self.max_hash_entries: int = max_hash_entries
        self.pattern_length: int = pattern_length
        self._hash_table: dict[tuple[int, ...], list[int]] = defaultdict(list)
        self._current_data: np.ndarray | None = None

    def initialize(self, data: np.ndarray) -> None:
        self._current_data = data
        self.clear_hash_table()

    def clear_hash_table(self) -> None:
        self._hash_table.clear()

    def update_hash_table(self, position: int) -> None:
        if self._current_data is None or position + self.pattern_length > len(
            self._current_data
        ):
            return

        pattern: tuple[int, ...] = tuple(
            self._current_data[position : position + self.pattern_length]
        )
        self._hash_table[pattern].append(position)

        # Limit list size to prevent excessive memory usage
        if len(self._hash_table[pattern]) > self.max_hash_entries:
            self._hash_table[pattern] = self._hash_table[pattern][
                -self.max_hash_entries :
            ]

    def find_longest_match_optimized(
        self,
        window: npt.NDArray[np.integer],
        lookahead: npt.NDArray[np.integer],
        current_absolute_pos: int,
    ) -> tuple[int, int]:
        n: int = window.size
        if n == 0 or lookahead.size == 0:
            return 0, 0

        best_d: int = 0
        max_l: int = 0

        # Try hash table optimization first
        if lookahead.size >= self.min_match_length:
            max_l, best_d = self._try_hash_matches(
                window, lookahead, current_absolute_pos, n
            )

        # Fallback to brute force if hash didn't find good matches
        if max_l < self.min_match_length:
            max_l, best_d = self._try_brute_force_matches(
                window, lookahead, max_l, best_d
            )

        return (best_d, max_l) if max_l > 0 else (0, 0)

    def _try_hash_matches(
        self,
        window: npt.NDArray[np.integer],
        lookahead: npt.NDArray[np.integer],
        current_absolute_pos: int,
        window_size: int,
    ) -> tuple[int, int]:
        max_l: int = 0
        best_d: int = 0

        pattern_length: int = min(self.pattern_length, lookahead.size)
        pattern: tuple[int, ...] = tuple(lookahead[:pattern_length])

        if pattern not in self._hash_table:
            return max_l, best_d

        # Check potential matches from hash table
        for match_pos in reversed(self._hash_table[pattern]):
            # Convert absolute position to relative position in window
            if (
                match_pos < current_absolute_pos - window_size
                or match_pos >= current_absolute_pos
            ):
                continue

            window_pos: int = match_pos - (current_absolute_pos - window_size)
            if window_pos < 0 or window_pos >= window_size:
                continue

            d: int = window_size - window_pos  # distance from current position
            if d <= 0:
                continue

            # Extend match with overlap support
            l: int = 0
            while l < lookahead.size and window[window_pos + (l % d)] == lookahead[l]:
                l += 1

            if l >= self.min_match_length and l > max_l:
                max_l = l
                best_d = d

                # Early termination if maximum possible match found
                if max_l == lookahead.size:
                    break

        return max_l, best_d

    def _try_brute_force_matches(
        self,
        window: npt.NDArray[np.integer],
        lookahead: npt.NDArray[np.integer],
        current_max_l: int,
        current_best_d: int,
    ) -> tuple[int, int]:
        """Fallback brute force matching."""
        max_l: int = current_max_l
        best_d: int = current_best_d

        # Try every start position in the window
        for start in range(window.size):
            d: int = window.size - start  # distance from current position
            if d <= 0:
                continue

            # Compare with overlap: the source is periodic with period d
            l: int = 0
            while l < lookahead.size and window[start + (l % d)] == lookahead[l]:
                l += 1

            if l > max_l:
                max_l = l
                best_d = d

            if max_l == lookahead.size:  # can't do better
                break

        return max_l, best_d
