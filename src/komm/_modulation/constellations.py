import numpy as np
import numpy.typing as npt


def constellation_pam(
    order: int,
    base_amplitude: float,
) -> npt.NDArray[np.float64]:
    return base_amplitude * np.arange(-order + 1, order, step=2, dtype=int)


def constellation_qam(
    orders: tuple[int, int],
    base_amplitudes: tuple[float, float],
    phase_offset: float,
) -> npt.NDArray[np.complex128]:
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
) -> npt.NDArray[np.complex128]:
    return base_amplitude * np.arange(order, dtype=int) * np.exp(1j * phase_offset)


def constellation_psk(
    order: int,
    amplitude: float,
    phase_offset: float,
) -> npt.NDArray[np.complex128]:
    return (
        amplitude
        * np.exp(2j * np.pi * np.arange(order) / order)
        * np.exp(1j * phase_offset)
    )


def constellation_apsk(
    orders: tuple[int, int],
    amplitudes: tuple[float, float],
    phase_offsets: tuple[float, float],
):
    return np.concatenate(
        [
            constellation_psk(M_k, A_k, phi_k)
            for M_k, A_k, phi_k in zip(orders, amplitudes, phase_offsets)
        ]
    )
