from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.conversations.models import ConversationMessageRole, ConversationMessageStatus
from app.core.config import settings
from app.rag.schemas import GroundedCitation


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    document_id: UUID | None = None
    version_id: UUID | None = None

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        title = value.strip()
        return title or None

    @model_validator(mode="after")
    def validate_scope(self) -> "ConversationCreateRequest":
        if self.version_id is not None and self.document_id is None:
            raise ValueError("version_id requires document_id")
        return self


class ConversationMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(max_length=settings.conversation_max_question_length)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    workspace_id: UUID
    title: str | None
    document_id: UUID | None
    version_id: UUID | None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    next_before_updated_at: datetime | None = None
    next_before_id: UUID | None = None


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sequence: int
    role: ConversationMessageRole
    content: str
    status: ConversationMessageStatus
    citations: list[GroundedCitation]
    insufficient_context: bool | None
    created_at: datetime


class ConversationDetailResponse(ConversationResponse):
    messages: list[ConversationMessageResponse]
    next_before_sequence: int | None = None


class ConversationMessagePairResponse(BaseModel):
    user_message: ConversationMessageResponse
    assistant_message: ConversationMessageResponse

