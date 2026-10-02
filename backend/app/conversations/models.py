from enum import StrEnum


class ConversationMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessageStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"