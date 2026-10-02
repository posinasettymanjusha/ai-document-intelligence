from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.conversations.models import ConversationMessageRole, ConversationMessageStatus
from app.db.models import (
    ConversationMessageRecord,
    ConversationRecord,
    WorkspaceMember,
)
from app.documents.repository import DocumentRepository


class ActiveConversationTurn(Exception):
    pass


class ConversationTurnNoLongerPending(Exception):
    pass


class ConversationRepository:
    def __init__(self, session: Session, document_repository: DocumentRepository) -> None:
        self._session = session
        self._document_repository = document_repository

    def is_workspace_member(self, workspace_id: UUID, user_id: UUID) -> bool:
        return self._document_repository.is_workspace_member(workspace_id, user_id)

    def validate_scope(
        self,
        workspace_id: UUID,
        user_id: UUID,
        document_id: UUID | None,
        version_id: UUID | None,
    ) -> bool:
        if version_id is not None and document_id is None:
            return False
        if document_id is None:
            return True

        document = self._document_repository.get_for_user(document_id, user_id)
        if document is None or document.workspace_id != workspace_id:
            return False
        return version_id is None or self._document_repository.user_can_access_version(
            document_id,
            version_id,
            user_id,
        )

    def create(
        self,
        workspace_id: UUID,
        owner_user_id: UUID,
        *,
        title: str | None,
        document_id: UUID | None,
        version_id: UUID | None,
    ) -> ConversationRecord:
        conversation = ConversationRecord(
            id=uuid4(),
            workspace_id=workspace_id,
            owner_user_id=owner_user_id,
            title=title,
            document_id=document_id,
            version_id=version_id,
        )
        self._session.add(conversation)
        self._session.commit()
        self._session.refresh(conversation)
        return conversation

    def list_for_owner(
        self,
        workspace_id: UUID,
        owner_user_id: UUID,
        *,
        limit: int,
        before_updated_at: datetime | None,
        before_id: UUID | None,
    ) -> list[ConversationRecord]:
        query = (
            select(ConversationRecord)
            .join(
                WorkspaceMember,
                WorkspaceMember.workspace_id == ConversationRecord.workspace_id,
            )
            .where(
                ConversationRecord.workspace_id == workspace_id,
                ConversationRecord.owner_user_id == owner_user_id,
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == owner_user_id,
            )
            .order_by(ConversationRecord.updated_at.desc(), ConversationRecord.id.desc())
            .limit(limit + 1)
        )
        if before_updated_at is not None and before_id is not None:
            query = query.where(
                or_(
                    ConversationRecord.updated_at < before_updated_at,
                    and_(
                        ConversationRecord.updated_at == before_updated_at,
                        ConversationRecord.id < before_id,
                    ),
                )
            )
        return list(self._session.scalars(query))

    def get_for_owner(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        *,
        for_update: bool = False,
    ) -> ConversationRecord | None:
        query = self._authorized_conversation_query(
            workspace_id,
            conversation_id,
            owner_user_id,
        )
        if for_update:
            query = query.with_for_update(of=ConversationRecord)
        return self._session.scalar(query)

    def messages_for_owner(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[ConversationMessageRecord]:
        query = self._authorized_message_query(
            workspace_id,
            conversation_id,
            owner_user_id,
        ).order_by(ConversationMessageRecord.sequence.desc()).limit(limit + 1)
        if before_sequence is not None:
            query = query.where(ConversationMessageRecord.sequence < before_sequence)
        messages = list(self._session.scalars(query))
        messages.reverse()
        return messages

    def begin_user_turn(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        question: str,
        *,
        max_history_turns: int,
        stale_pending_seconds: int,
    ) -> tuple[ConversationRecord, ConversationMessageRecord, list[ConversationMessageRecord]] | None:
        try:
            conversation = self.get_for_owner(
                workspace_id,
                conversation_id,
                owner_user_id,
                for_update=True,
            )
            if conversation is None:
                self._session.rollback()
                return None

            pending_query = (
                self._authorized_message_query(workspace_id, conversation_id, owner_user_id)
                .where(
                    ConversationMessageRecord.role == ConversationMessageRole.USER.value,
                    ConversationMessageRecord.status == ConversationMessageStatus.PENDING.value,
                )
                .order_by(ConversationMessageRecord.sequence.desc())
                .limit(1)
            )
            pending = self._session.scalar(pending_query)
            now = datetime.now(UTC)
            if pending is not None:
                started_at = pending.created_at
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=UTC)
                if now - started_at < timedelta(seconds=stale_pending_seconds):
                    self._session.rollback()
                    raise ActiveConversationTurn
                pending.status = ConversationMessageStatus.FAILED.value

            history_query = (
                self._authorized_message_query(workspace_id, conversation_id, owner_user_id)
                .where(
                    ConversationMessageRecord.status == ConversationMessageStatus.COMPLETE.value,
                )
                .order_by(ConversationMessageRecord.sequence.desc())
                .limit(max_history_turns * 2)
            )
            history = list(self._session.scalars(history_query))
            history.reverse()

            next_sequence = self._next_sequence(
                workspace_id,
                conversation_id,
                owner_user_id,
            )
            user_message = ConversationMessageRecord(
                id=uuid4(),
                conversation_id=conversation_id,
                sequence=next_sequence,
                role=ConversationMessageRole.USER.value,
                content=question,
                status=ConversationMessageStatus.PENDING.value,
                citations=[],
                insufficient_context=None,
            )
            conversation.updated_at = now
            self._session.add(user_message)
            self._session.commit()
            self._session.refresh(conversation)
            self._session.refresh(user_message)
            return conversation, user_message, history
        except ActiveConversationTurn:
            raise
        except Exception:
            self._session.rollback()
            raise

    def complete_user_turn(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        user_message_id: UUID,
        *,
        answer: str,
        citations: list[dict[str, object]],
        insufficient_context: bool,
    ) -> tuple[ConversationMessageRecord, ConversationMessageRecord] | None:
        try:
            conversation = self.get_for_owner(
                workspace_id,
                conversation_id,
                owner_user_id,
                for_update=True,
            )
            if conversation is None:
                self._session.rollback()
                return None

            user_message = self._session.scalar(
                self._authorized_message_query(workspace_id, conversation_id, owner_user_id).where(
                    ConversationMessageRecord.id == user_message_id,
                    ConversationMessageRecord.role == ConversationMessageRole.USER.value,
                )
            )
            if (
                user_message is None
                or user_message.status != ConversationMessageStatus.PENDING.value
            ):
                self._session.rollback()
                raise ConversationTurnNoLongerPending

            assistant_message = ConversationMessageRecord(
                id=uuid4(),
                conversation_id=conversation_id,
                sequence=self._next_sequence(
                    workspace_id,
                    conversation_id,
                    owner_user_id,
                ),
                role=ConversationMessageRole.ASSISTANT.value,
                content=answer,
                status=ConversationMessageStatus.COMPLETE.value,
                citations=citations,
                insufficient_context=insufficient_context,
            )
            user_message.status = ConversationMessageStatus.COMPLETE.value
            conversation.updated_at = datetime.now(UTC)
            self._session.add(assistant_message)
            self._session.commit()
            self._session.refresh(user_message)
            self._session.refresh(assistant_message)
            return user_message, assistant_message
        except ConversationTurnNoLongerPending:
            raise
        except Exception:
            self._session.rollback()
            raise

    def mark_user_turn_failed(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        user_message_id: UUID,
    ) -> None:
        try:
            conversation = self.get_for_owner(
                workspace_id,
                conversation_id,
                owner_user_id,
                for_update=True,
            )
            if conversation is None:
                self._session.rollback()
                return
            user_message = self._session.scalar(
                self._authorized_message_query(workspace_id, conversation_id, owner_user_id).where(
                    ConversationMessageRecord.id == user_message_id,
                    ConversationMessageRecord.role == ConversationMessageRole.USER.value,
                )
            )
            if (
                user_message is not None
                and user_message.status == ConversationMessageStatus.PENDING.value
            ):
                user_message.status = ConversationMessageStatus.FAILED.value
                self._session.commit()
            else:
                self._session.rollback()
        except Exception:
            self._session.rollback()
            raise

    def delete_for_owner(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
    ) -> bool:
        conversation = self.get_for_owner(
            workspace_id,
            conversation_id,
            owner_user_id,
            for_update=True,
        )
        if conversation is None:
            self._session.rollback()
            return False
        self._session.delete(conversation)
        self._session.commit()
        return True

    def rollback(self) -> None:
        self._session.rollback()

    def _authorized_conversation_query(self, workspace_id, conversation_id, owner_user_id):
        return (
            select(ConversationRecord)
            .join(
                WorkspaceMember,
                WorkspaceMember.workspace_id == ConversationRecord.workspace_id,
            )
            .where(
                ConversationRecord.workspace_id == workspace_id,
                ConversationRecord.id == conversation_id,
                ConversationRecord.owner_user_id == owner_user_id,
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == owner_user_id,
            )
        )

    def _authorized_message_query(self, workspace_id, conversation_id, owner_user_id):
        return (
            select(ConversationMessageRecord)
            .join(
                ConversationRecord,
                ConversationRecord.id == ConversationMessageRecord.conversation_id,
            )
            .join(
                WorkspaceMember,
                WorkspaceMember.workspace_id == ConversationRecord.workspace_id,
            )
            .where(
                ConversationRecord.workspace_id == workspace_id,
                ConversationRecord.id == conversation_id,
                ConversationRecord.owner_user_id == owner_user_id,
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == owner_user_id,
            )
        )

    def _next_sequence(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
    ) -> int:
        current = self._session.scalar(
            self._authorized_message_query(
                workspace_id,
                conversation_id,
                owner_user_id,
            ).with_only_columns(func.max(ConversationMessageRecord.sequence))
        )
        return int(current or 0) + 1