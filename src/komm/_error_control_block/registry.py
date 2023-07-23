from typing import Callable, Literal, TypedDict

import numpy as np

Decoder = Callable[..., np.ndarray]  # TODO: improve typing


class BlockDecoderData(TypedDict):
    description: str
    decoder: Decoder
    type_in: Literal["soft", "hard"]
    type_out: Literal["soft", "hard"]
    target: Literal["codeword", "message"]


class RegistryBlockDecoder:
    _registry = {}

    @classmethod
    def register(cls, method: str, data: BlockDecoderData) -> None:
        cls._registry[method] = data

    @classmethod
    def is_registered(cls, method: str) -> bool:
        return method in cls._registry

    @classmethod
    def get(cls, method: str) -> BlockDecoderData:
        return cls._registry[method]
