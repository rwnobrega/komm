import functools
import inspect

import numpy as np


def must_be_pmf(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1:
        raise ValueError("Must be a 1D-array")
    if not np.all(value >= 0.0):
        raise ValueError("All elements must be non-negative")
    if not np.isclose(value.sum(), 1.0):
        raise ValueError("The sum of all elements must be 1.0")
    return value


def must_be_log_base(value):
    if isinstance(value, str) and value != "e":
        raise ValueError("If string, must be 'e'")
    if isinstance(value, float) and value <= 0.0:
        raise ValueError("If float, must be positive")
    return value


def must_be_in_set(set):
    def validator(value):
        if value not in set:
            raise ValueError(f"Must be in {set}")
        return value

    return validator


def must_be_at_least(min):
    def validator(value):
        if value < min:
            raise ValueError(f"Must be at least {min}")
        return value

    return validator


def validate(**validators):
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for arg, validator in validators.items():
                if arg in bound.arguments:
                    try:
                        bound.arguments[arg] = validator(bound.arguments[arg])
                    except ValueError as e:
                        raise ValueError(f"Invalid value for argument '{arg}': {e}")
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
