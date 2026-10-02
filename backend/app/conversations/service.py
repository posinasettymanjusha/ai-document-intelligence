from collections.abc import Sequence
import re
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.conversations.models import ConversationMessageRole, ConversationMessageStatus
from app.conversations.repository import (
    ActiveConversationTurn,
    ConversationRepository,
    ConversationTurnNoLongerPending,
)
from app.conversations.schemas import (
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationMessagePairResponse,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationResponse,
)
from app.core.config import settings
from app.core.errors import AppError
from app.rag.schemas import ConversationContextMessage, GroundedAnswerRequest
from app.rag.service import GroundedAnswerService

_HISTORICAL_SOURCE_LABEL = re.compile(r"\[S\d+\]")


class ConversationService:
    def __init__(
        self,
        repository: ConversationRepository,
        grounded_answer_service: GroundedAnswerService,
    ) -> None:
        self._repository = repository
        self._grounded_answer_service = grounded_answer_service

    def create(
        self,
        workspace_id: UUID,
        user_id: str,
        request: ConversationCreateRequest,
    ) -> ConversationResponse:
        user_uuid = self._user_uuid(user_id)
        try:
            if not self._repository.is_workspace_member(workspace_id, user_uuid):
                self._not_found()
            if not self._repository.validate_scope(
                workspace_id,
                user_uuid,
                request.document_id,
                request.version_id,
            ):
                self._not_found()
            conversation = self._repository.create(
                workspace_id,
                user_uuid,
                title=request.title,
                document_id=request.document_id,
                version_id=request.version_id,
            )
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "The conversation could not be created.",
            ) from error
        return ConversationResponse.model_validate(conversation)

    def list_conversations(
        self,
        workspace_id: UUID,
        user_id: str,
        *,
        limit: int,
        before_updated_at=None,
        before_id: UUID | None = None,
    ) -> ConversationListResponse:
        user_uuid = self._user_uuid(user_id)
        try:
            if not self._repository.is_workspace_member(workspace_id, user_uuid):
                self._not_found()
            conversations = self._repository.list_for_owner(
                workspace_id,
                user_uuid,
                limit=limit,
                before_updated_at=before_updated_at,
                before_id=before_id,
            )
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "Conversations could not be loaded.",
            ) from error
        has_more = len(conversations) > limit
        page = conversations[:limit]
        if page:
            cursor_updated_at = page[-1].updated_at if has_more else None
            cursor_id = page[-1].id if has_more else None
        else:
            cursor_updated_at = None
            cursor_id = None
        return ConversationListResponse(
            items=[ConversationResponse.model_validate(item) for item in page],
            next_before_updated_at=cursor_updated_at,
            next_before_id=cursor_id,
        )

    def get(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        user_id: str,
        *,
        message_limit: int,
        before_sequence: int | None = None,
    ) -> ConversationDetailResponse:
        user_uuid = self._user_uuid(user_id)
        try:
            conversation = self._repository.get_for_owner(
                workspace_id,
                conversation_id,
                user_uuid,
            )
            if conversation is None:
                self._not_found()
            messages = self._repository.messages_for_owner(
                workspace_id,
                conversation_id,
                user_uuid,
                limit=message_limit,
                before_sequence=before_sequence,
            )
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "The conversation could not be loaded.",
            ) from error
        has_more = len(messages) > message_limit
        page = messages[-message_limit:]
        next_before_sequence = page[0].sequence if has_more and page else None
        return ConversationDetailResponse(
            **ConversationResponse.model_validate(conversation).model_dump(),
            messages=[self._message_response(message) for message in page],
            next_before_sequence=next_before_sequence,
        )

    def send_message(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        user_id: str,
        request: ConversationMessageRequest,
    ) -> ConversationMessagePairResponse:
        user_uuid = self._user_uuid(user_id)
        try:
            begun = self._repository.begin_user_turn(
                workspace_id,
                conversation_id,
                user_uuid,
                request.question,
                max_history_turns=settings.conversation_max_history_turns,
                stale_pending_seconds=settings.conversation_stale_pending_seconds,
            )
        except ActiveConversationTurn as error:
            raise AppError(
                409,
                "conversation_turn_in_progress",
                "A message is already being processed for this conversation.",
            ) from error
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "The user message could not be saved.",
            ) from error
        if begun is None:
            self._not_found()
        conversation, user_message, history_records = begun

        history = self._bounded_history(history_records)
        retrieval_query = self._retrieval_query(request.question, history)
        try:
            grounded_answer = self._grounded_answer_service.answer_with_context(
                workspace_id,
                str(user_uuid),
                GroundedAnswerRequest(
                    question=request.question,
                    top_k=request.top_k,
                    document_id=conversation.document_id,
                    version_id=conversation.version_id,
                ),
                conversation_history=history,
                retrieval_query=retrieval_query,
            )
        except Exception as error:
            self._mark_failed(
                workspace_id,
                conversation_id,
                user_uuid,
                user_message.id,
            )
            if isinstance(error, AppError):
                raise
            raise AppError(
                503,
                "answer_unavailable",
                "An answer could not be generated right now.",
            ) from error

        try:
            persisted = self._repository.complete_user_turn(
                workspace_id,
                conversation_id,
                user_uuid,
                user_message.id,
                answer=grounded_answer.answer,
                citations=[citation.model_dump(mode="json") for citation in grounded_answer.citations],
                insufficient_context=grounded_answer.insufficient_context,
            )
        except ConversationTurnNoLongerPending as error:
            raise AppError(
                409,
                "conversation_turn_expired",
                "The conversation turn expired before its answer could be saved.",
            ) from error
        except SQLAlchemyError as error:
            self._repository.rollback()
            self._mark_failed(workspace_id, conversation_id, user_uuid, user_message.id)
            raise AppError(
                503,
                "persistence_unavailable",
                "The answer could not be saved.",
            ) from error
        if persisted is None:
            self._not_found()
        return ConversationMessagePairResponse(
            user_message=self._message_response(persisted[0]),
            assistant_message=self._message_response(persisted[1]),
        )

    def delete(self, workspace_id: UUID, conversation_id: UUID, user_id: str) -> None:
        user_uuid = self._user_uuid(user_id)
        try:
            deleted = self._repository.delete_for_owner(
                workspace_id,
                conversation_id,
                user_uuid,
            )
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "The conversation could not be deleted.",
            ) from error
        if not deleted:
            self._not_found()

    def _mark_failed(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        user_uuid: UUID,
        user_message_id: UUID,
    ) -> None:
        try:
            self._repository.mark_user_turn_failed(
                workspace_id,
                conversation_id,
                user_uuid,
                user_message_id,
            )
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "The failed message state could not be saved.",
            ) from error

    @staticmethod
    def _bounded_history(records: Sequence) -> list[ConversationContextMessage]:
        turns: list[tuple[object, object]] = []
        index = 0
        while index + 1 < len(records):
            user_message = records[index]
            assistant_message = records[index + 1]
            if (
                user_message.role == ConversationMessageRole.USER.value
                and assistant_message.role == ConversationMessageRole.ASSISTANT.value
            ):
                turns.append((user_message, assistant_message))
                index += 2
            else:
                index += 1

        maximum_characters = settings.conversation_max_history_characters
        selected: list[tuple[str, str]] = []
        used_characters = 0
        for user_message, assistant_message in reversed(turns[-settings.conversation_max_history_turns :]):
            user_content = user_message.content
            assistant_content = _HISTORICAL_SOURCE_LABEL.sub("", assistant_message.content)
            full_length = len(user_content) + len(assistant_content)
            remaining = maximum_characters - used_characters
            if full_length <= remaining:
                selected.append((user_content, assistant_content))
                used_characters += full_length
                continue

            if not selected and remaining > 0:
                bounded_user = user_content[:remaining]
                remaining -= len(bounded_user)
                bounded_assistant = assistant_content[:remaining]
                selected.append((bounded_user, bounded_assistant))
            break

        history: list[ConversationContextMessage] = []
        for user_text, assistant_text in reversed(selected):
            history.extend(
                [
                    ConversationContextMessage(
                        role=ConversationMessageRole.USER.value,
                        content=user_text,
                    ),
                    ConversationContextMessage(
                        role=ConversationMessageRole.ASSISTANT.value,
                        content=assistant_text,
                    ),
                ]
            )
        return history

    @staticmethod
    def _retrieval_query(
        question: str,
        history: Sequence[ConversationContextMessage],
    ) -> str:
        if not history:
            return question
        context_lines = [f"{message.role}: {message.content}" for message in history]
        return "Recent conversation context:\n" + "\n".join(context_lines) + f"\nCurrent question: {question}"

    @staticmethod
    def _message_response(message) -> ConversationMessageResponse:
        return ConversationMessageResponse.model_validate(message)

    @staticmethod
    def _user_uuid(user_id: str) -> UUID:
        try:
            return UUID(user_id)
        except ValueError as error:
            raise AppError(401, "unauthorized", "A valid user identity is required.") from error

    @staticmethod
    def _not_found() -> None:
        raise AppError(404, "not_found", "The requested resource was not found.")