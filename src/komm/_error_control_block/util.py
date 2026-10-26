import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .._finite_state_machine.Trellis import Trellis
from .._finite_state_machine.TrellisSection import TrellisSection
from .._util.matrices import row_span, trellis_oriented_form
from ..types import Array2D


def extended_parity_submatrix(
    parity_submatrix: Array2D[np.integer],
) -> Array2D[np.integer]:
    last_column = (1 + np.sum(parity_submatrix, axis=1)) % 2
    extended_parity_submatrix = np.hstack([parity_submatrix, last_column[np.newaxis].T])
    return extended_parity_submatrix


def span_weight_distribution(matrix: Array2D[np.integer]) -> list[int]:
    n = matrix.shape[1]
    # Packed bits: faster XOR and popcount.
    packed = np.packbits(matrix, axis=1)
    # Batches of up to 2**10 words.
    low, high = row_span(packed[:10]), row_span(packed[10:])
    distribution = np.zeros(n + 1, dtype=int)
    desc = "Computing codeword weight distribution"
    for offset in tqdm(high, desc=desc, delay=2.5, unit_scale=len(low)):
        weights = np.bitwise_count(low ^ offset).sum(axis=1, dtype=int)
        distribution += np.bincount(weights, minlength=n + 1)
    return distribution.tolist()


def macwilliams_transform(distribution: list[int]) -> list[int]:
    # Weight distribution of the dual code.
    n, size = len(distribution) - 1, sum(distribution)
    dual = [0] * (n + 1)
    for i, a in enumerate(distribution):
        if a == 0:
            continue
        # Coefficients of (1 - z)^i (1 + z)^(n - i).
        prev, curr = 0, 1
        for j in range(n + 1):
            dual[j] += a * curr
            prev, curr = curr, ((n - 2 * i) * curr - (n - j + 1) * prev) // (j + 1)
    return [b // size for b in dual]


def syndrome_trellis(check_matrix: npt.ArrayLike) -> Trellis:
    # States: partial syndromes of active rows
    tof = trellis_oriented_form(check_matrix)
    n = tof.shape[1]
    first = np.argmax(tof, axis=1)
    last = n - 1 - np.argmax(tof[:, ::-1], axis=1)
    x = np.array([0, 1])  # Code bit, the single label
    sections: list[TrellisSection] = []
    for t in range(n):
        # Rows active at time t, or starting
        rows = np.flatnonzero((first <= t) & (t <= last))
        active, ending = first[rows] < t, last[rows] == t
        num_active = int(np.count_nonzero(active))
        num_states = 2**num_active
        bits = np.zeros((num_states, rows.size), dtype=int)
        bits[:, active] = (np.arange(num_states)[:, None] >> np.arange(num_active)) & 1
        # Partial syndromes after bit x
        syndromes = bits[:, None, :] ^ (tof[rows, t] * x[:, None])
        valid = ~syndromes[..., ending].any(axis=-1)
        weights = 1 << np.arange(np.count_nonzero(~ending))
        transitions = np.where(valid, syndromes[..., ~ending] @ weights, -1)
        sections.append(TrellisSection(transitions))
    return Trellis(sections)
