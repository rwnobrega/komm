import functools
import inspect
from typing import Any, Callable, ParamSpec, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from attrs import asdict, make_class


class Attr(Protocol):
    name: str


def is_probability(inst: object, attr: Attr, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"'{attr.name}' must be between 0 and 1")


def is_pmf(inst: object, attr: Attr, value: npt.NDArray[np.float64]) -> None:
    if value.ndim != 1:
        raise ValueError(f"'{attr.name}' must be a 1D array")
    if not np.all(value >= 0.0):
        raise ValueError(f"'{attr.name}' must be non-negative")
    if not np.isclose(value.sum(), 1.0):
        raise ValueError(f"'{attr.name}' must sum to 1.0")


def is_transition_matrix(
    inst: object, attr: Attr, value: npt.NDArray[np.float64]
) -> None:
    if value.ndim != 2:
        raise ValueError(f"'{attr.name}' must be a 2D array")
    if not np.all(value >= 0.0):
        raise ValueError(f"'{attr.name}' must be non-negative")
    if not np.allclose(value.sum(axis=1), 1.0):
        raise ValueError(f"Rows of '{attr.name}' must sum to 1.0")


def is_log_base(inst: object, attr: Attr, value: float | str):
    if (isinstance(value, str) and value != "e") or (
        isinstance(value, float) and value <= 0.0
    ):
        raise ValueError(f"'{attr.name}' must be 'e' or a positive float")


P = ParamSpec("P")
R = TypeVar("R")


def validate_call(**fields: Any):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            Params = make_class("Params", fields)
            bound_args = {k: v for k, v in bound.arguments.items() if k in fields}
            inst = Params(**bound_args)  # Use attrs to validate and convert args
            bound.arguments.update(asdict(inst))
            return func(**bound.arguments)  # type: ignore

        return wrapper

    return decorator
