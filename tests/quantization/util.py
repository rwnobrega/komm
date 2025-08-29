import numpy as np

uniform_pdf = lambda x, peak: 1 / (2 * peak) * (np.abs(x) <= peak)
gaussian_pdf = lambda x: 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)
laplacian_pdf = lambda x: 1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.abs(x))
