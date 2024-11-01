import functools
import inspect

import numpy as np
from attrs import asdict, make_class


def is_pmf(inst, attr, value: np.ndarray):
    if value.ndim != 1:
        raise ValueError(f"'{attr.name}' must be a 1D array")
    if not np.all(value >= 0.0):
        raise ValueError(f"'{attr.name}' must be non-negative")
    if not np.isclose(value.sum(), 1.0):
        raise ValueError(f"'{attr.name}' must sum to 1.0")


def is_log_base(inst, attr, value: float | str):
    if (isinstance(value, str) and value != "e") or (
        isinstance(value, float) and value <= 0.0
    ):
        raise ValueError(f"'{attr.name}' must be 'e' or a positive float")


def validate_call(**fields):
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            Params = make_class("Params", fields)
            bound_args = {k: v for k, v in bound.arguments.items() if k in fields}
            inst = Params(**bound_args)  # Use attrs to validate and convert args
            bound.arguments.update(asdict(inst))
            return func(**bound.arguments)

        return wrapper

    return decorator
