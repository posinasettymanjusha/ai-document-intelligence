import math

from app.core.config import settings
from app.embeddings.models import EmbeddingResult
from app.embeddings.providers import EmbeddingProvider


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        expected_dimension: int | None = None,
    ) -> None:
        self._provider = provider
        self._expected_dimension = (
            expected_dimension if expected_dimension is not None else settings.embedding_dimension
        )
        if provider.dimension != self._expected_dimension:
            raise ValueError(
                f"Provider dimension {provider.dimension} does not match configured dimension "
                f"{self._expected_dimension}"
            )

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    @property
    def dimension(self) -> int:
        return self._expected_dimension

    def embed_text(self, text: str) -> EmbeddingResult:
        vector = self._provider.embed_text(text)
        self._validate_vector(vector)
        return EmbeddingResult(model_id=self._provider.model_id, vector=vector)

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        vectors = self._provider.embed_texts(texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedding provider returned a different number of vectors than inputs")
        for vector in vectors:
            self._validate_vector(vector)
        return [EmbeddingResult(model_id=self._provider.model_id, vector=vector) for vector in vectors]

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._expected_dimension:
            raise ValueError(
                f"Embedding has dimension {len(vector)}; expected {self._expected_dimension}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding values must all be finite numbers")