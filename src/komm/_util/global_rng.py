import numpy as np

_rng = np.random.default_rng()


def get():
    return _rng


def set(rng: np.random.Generator):
    global _rng
    _rng = rng
