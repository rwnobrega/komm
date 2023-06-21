# These are unused functions that could be useful in the future

import numpy as np


def uniform_real_hard_demodulator(received, order):
    return np.clip(np.around((received + order - 1) / 2), 0, order - 1).astype(int)


def uniform_real_soft_bit_demodulator(received, snr):
    return -4 * snr * received


def ask_hard_demodulator(received, order):
    return np.clip(np.around((received.real + order - 1) / 2), 0, order - 1).astype(int)


def psk_hard_demodulator(received, order):
    phase_in_turns = np.angle(received) / (2 * np.pi)
    return np.mod(np.around(phase_in_turns * order).astype(int), order)


def bpsk_soft_bit_demodulator(received, snr):
    return 4 * snr * received.real


def qpsk_soft_bit_demodulator_reflected(received, snr):
    received_rotated = received * np.exp(2j * np.pi / 8)
    soft_bits = np.empty(2 * received.size, dtype=float)
    soft_bits[0::2] = np.sqrt(8) * snr * received_rotated.real
    soft_bits[1::2] = np.sqrt(8) * snr * received_rotated.imag
    return soft_bits


def rectangular_hard_demodulator(received, order):
    L = int(np.sqrt(order))
    s_real = np.clip(np.around((received.real + L - 1) / 2), 0, L - 1).astype(int)
    s_imag = np.clip(np.around((received.imag + L - 1) / 2), 0, L - 1).astype(int)
    return s_real + L * s_imag
