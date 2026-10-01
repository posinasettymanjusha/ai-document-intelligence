from dataclasses import dataclass
from enum import StrEnum


class EmbeddingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class EmbeddingResult:
    model_id: str
    vector: list[float]

    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass(frozen=True)
class EmbeddingGenerationSummary:
    attempted_chunks: int
    ready_chunks: int
    failed_chunks: int
    failure_message: str | None = None