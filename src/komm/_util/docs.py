from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def mkdocstrings(**options: Any) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        def _doc_options(cls: type[T]) -> dict[str, Any]:
            return options

        setattr(cls, "_doc_options", classmethod(_doc_options))
        return cls

    return decorator
