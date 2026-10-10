import numpy as np


def uniform_pdf(x, peak):
    return 1 / (2 * peak) * (np.abs(x) <= peak)


def gaussian_pdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)


def laplacian_pdf(x):
    return 1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.abs(x))
