from typing import Any


def format_list_no_quotes(a: Any | list[Any]) -> str:
    if isinstance(a, list):
        return "[" + ", ".join(format_list_no_quotes(x) for x in a) + "]"  # type: ignore
    return str(a)
