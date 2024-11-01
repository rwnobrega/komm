import numpy as np


def constellation_pam(order, base_amplitude):
    return base_amplitude * np.arange(-order + 1, order, step=2, dtype=int)


def constellation_qam(orders, base_amplitudes, phase_offset):
    order_I, order_Q = orders
    base_amplitude_I, base_amplitude_Q = base_amplitudes
    constellation_I = constellation_pam(order_I, base_amplitude_I)
    constellation_Q = constellation_pam(order_Q, base_amplitude_Q)
    return (constellation_I + 1j * constellation_Q[np.newaxis].T).flatten() * np.exp(
        1j * phase_offset
    )


def constellation_ask(order, base_amplitude, phase_offset):
    return base_amplitude * np.arange(order, dtype=int) * np.exp(1j * phase_offset)


def constellation_psk(order, amplitude, phase_offset):
    return (
        amplitude
        * np.exp(2j * np.pi * np.arange(order) / order)
        * np.exp(1j * phase_offset)
    )


def constellation_apsk(orders, amplitudes, phase_offsets):
    return np.concatenate(
        [
            constellation_psk(M_k, A_k, phi_k)
            for M_k, A_k, phi_k in zip(orders, amplitudes, phase_offsets)
        ]
    )
