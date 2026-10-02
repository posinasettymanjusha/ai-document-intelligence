from uuid import UUID

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GroundedAnswerRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=10)
    document_id: UUID | None = None
    version_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class GroundedCitation(BaseModel):
    source_id: str
    document_id: UUID
    filename: str
    version_id: UUID
    version_number: int
    chunk_id: UUID
    chunk_index: int
    page_numbers: list[int] = Field(default_factory=list)
    source_metadata: dict[str, object]


class GroundedAnswerResponse(BaseModel):
    answer: str
    insufficient_context: bool
    citations: list[GroundedCitation]


class ConversationContextMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    insufficient_context: bool

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        answer = value.strip()
        if not answer:
            raise ValueError("answer must not be blank")
        return answer