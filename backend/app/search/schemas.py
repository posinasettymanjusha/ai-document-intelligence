from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    document_id: UUID | None = None
    version_id: UUID | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


class SemanticSearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    version_id: UUID
    version_number: int
    chunk_index: int
    text: str
    cosine_distance: float
    similarity: float
    source_metadata: dict[str, object]