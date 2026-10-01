from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_index: int
    text: str
    character_count: int
    estimated_token_count: int
    source_metadata: dict[str, object]