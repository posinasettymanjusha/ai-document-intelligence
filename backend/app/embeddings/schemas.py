from pydantic import BaseModel, Field


class EmbeddingGenerationRequest(BaseModel):
    retry_failed: bool = False


class EmbeddingGenerationResponse(BaseModel):
    attempted_chunks: int = Field(ge=0)
    ready_chunks: int = Field(ge=0)
    failed_chunks: int = Field(ge=0)
    failure_message: str | None = None