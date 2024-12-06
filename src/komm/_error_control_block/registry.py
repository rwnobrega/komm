from collections.abc import Callable
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt

Decoder = Callable[..., npt.NDArray[np.int_ | np.float64]]


class BlockDecoderData(TypedDict):
    description: str
    decoder: Decoder
    type_in: Literal["soft", "hard"]
    type_out: Literal["soft", "hard"]
    target: Literal["codeword", "message"]


class RegistryBlockDecoder:
    _registry: dict[str, BlockDecoderData] = {}

    @classmethod
    def register(cls, method: str, data: BlockDecoderData) -> None:
        cls._registry[method] = data

    @classmethod
    def is_registered(cls, method: str) -> bool:
        return method in cls._registry

    @classmethod
    def get(cls, method: str) -> BlockDecoderData:
        return cls._registry[method]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())
