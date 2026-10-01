import hashlib
import math
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_text(self, text: str) -> list[float]: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicHashEmbeddingProvider:
    """Small deterministic provider for plumbing tests; vectors are not semantic."""

    model_id = "local-hash-v1"

    def __init__(self, dimension: int = 768) -> None:
        if dimension < 1 or dimension > 2000:
            raise ValueError("dimension must be between 1 and 2000 for the configured vector index")
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.casefold())
        if not tokens:
            raise ValueError("Cannot embed empty or whitespace-only text")

        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[0] = 1.0
            norm = 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]