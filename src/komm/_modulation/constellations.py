import numpy as np

from ..types import Array1D


def constellation_pam(
    order: int,
    base_amplitude: float,
) -> Array1D[np.floating]:
    return base_amplitude * np.arange(-order + 1, order, step=2, dtype=int)


def constellation_qam(
    orders: tuple[int, int],
    base_amplitudes: tuple[float, float],
    phase_offset: float,
) -> Array1D[np.complexfloating]:
    order_I, order_Q = orders
    base_amplitude_I, base_amplitude_Q = base_amplitudes
    constellation_I = constellation_pam(order_I, base_amplitude_I)
    constellation_Q = constellation_pam(order_Q, base_amplitude_Q)
    constellation = (constellation_I + 1j * constellation_Q[np.newaxis].T).flatten()
    constellation *= np.exp(1j * phase_offset)
    return constellation


def constellation_ask(
    order: int,
    base_amplitude: float,
    phase_offset: float,
) -> Array1D[np.complexfloating]:
    return base_amplitude * np.arange(order, dtype=int) * np.exp(1j * phase_offset)


def constellation_psk(
    order: int,
    amplitude: float,
    phase_offset: float,
) -> Array1D[np.complexfloating]:
    # We round to avoid sin(pi) != 0
    return (
        amplitude
        * np.exp(2j * np.pi * np.arange(order) / order)
        * np.exp(1j * phase_offset)
    ).round(15)


def constellation_apsk(
    orders: tuple[int, ...],
    amplitudes: tuple[float, ...],
    phase_offsets: tuple[float, ...],
) -> Array1D[np.complexfloating]:
    return np.concatenate([
        constellation_psk(order, amplitude, phase_offset)
        for order, amplitude, phase_offset in zip(orders, amplitudes, phase_offsets)
    ])
