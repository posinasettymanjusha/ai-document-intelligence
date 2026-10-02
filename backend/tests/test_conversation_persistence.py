from unittest.mock import MagicMock
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.db.models import (
    ConversationMessageRecord,
    ConversationRecord,
)
from app.documents.repository import DocumentRepository
from app.conversations.repository import ConversationRepository
from app.core.config import Settings

WORKSPACE_ID = UUID("17d67f65-81f3-49db-a199-6357cd8f46f9")
USER_ID = UUID("6aa5a694-34c4-44f9-a631-4d9e11f27d1e")
CONVERSATION_ID = UUID("aca07028-48ec-4177-9a5b-6710e11df2ce")


def compile_statement(statement) -> tuple[str, dict[str, object]]:
    compiled = statement.compile(dialect=postgresql.dialect())
    return str(compiled), compiled.params


def test_conversation_queries_always_filter_workspace_conversation_owner_and_membership() -> None:
    session = MagicMock(spec=Session)
    repository = ConversationRepository(session, MagicMock())  # type: ignore[arg-type]

    conversation_sql, conversation_params = compile_statement(
        repository._authorized_conversation_query(WORKSPACE_ID, CONVERSATION_ID, USER_ID)
    )
    message_sql, message_params = compile_statement(
        repository._authorized_message_query(WORKSPACE_ID, CONVERSATION_ID, USER_ID)
    )

    for sql, params in [
        (conversation_sql, conversation_params),
        (message_sql, message_params),
    ]:
        assert "conversations.workspace_id =" in sql
        assert "conversations.id =" in sql
        assert "conversations.owner_user_id =" in sql
        assert "workspace_members.workspace_id =" in sql
        assert "workspace_members.user_id =" in sql
        assert WORKSPACE_ID in params.values()
        assert CONVERSATION_ID in params.values()
        assert USER_ID in params.values()


def test_conversation_row_lock_is_limited_to_conversation_record() -> None:
    session = MagicMock(spec=Session)
    repository = ConversationRepository(session, MagicMock())  # type: ignore[arg-type]
    statement = repository._authorized_conversation_query(
        WORKSPACE_ID,
        CONVERSATION_ID,
        USER_ID,
    ).with_for_update(of=ConversationRecord)

    sql, _ = compile_statement(statement)

    assert "FOR UPDATE OF conversations" in sql


def test_message_sequence_is_computed_from_existing_rows_not_mutable_conversation_state() -> None:
    session = MagicMock(spec=Session)
    session.scalar.return_value = 9
    repository = ConversationRepository(session, MagicMock())  # type: ignore[arg-type]

    next_sequence = repository._next_sequence(WORKSPACE_ID, CONVERSATION_ID, USER_ID)
    statement = session.scalar.call_args.args[0]
    sql, parameters = compile_statement(statement)

    assert next_sequence == 10
    assert "max(conversation_messages.sequence)" in sql
    assert "conversations.workspace_id =" in sql
    assert "conversations.id =" in sql
    assert "conversations.owner_user_id =" in sql
    assert "workspace_members.user_id =" in sql
    assert CONVERSATION_ID in parameters.values()


def test_conversation_schema_has_workspace_owner_and_scoped_document_foreign_keys() -> None:
    foreign_keys = {
        constraint.name: constraint
        for constraint in ConversationRecord.__table__.foreign_key_constraints
    }

    assert foreign_keys["fk_conversations_workspace_document"].ondelete == "RESTRICT"
    assert foreign_keys["fk_conversations_document_version"].ondelete == "RESTRICT"
    assert any(
        foreign_key.target_fullname == "workspaces.id"
        and foreign_key.ondelete == "CASCADE"
        for constraint in ConversationRecord.__table__.foreign_key_constraints
        for foreign_key in constraint.elements
    )
    assert any(
        foreign_key.target_fullname == "profiles.id"
        and foreign_key.ondelete == "CASCADE"
        for constraint in ConversationRecord.__table__.foreign_key_constraints
        for foreign_key in constraint.elements
    )


def test_message_schema_has_ordering_cascade_and_json_citations() -> None:
    constraints = list(ConversationMessageRecord.__table__.constraints)
    indexes = list(ConversationMessageRecord.__table__.indexes)

    assert any(
        constraint.name == "uq_conversation_messages_conversation_sequence"
        for constraint in constraints
    )
    assert any(
        foreign_key.ondelete == "CASCADE"
        for constraint in constraints
        for foreign_key in getattr(constraint, "elements", [])
    )
    assert ConversationMessageRecord.__table__.c.citations.type.__class__.__name__ == "JSONB"
    assert any(
        index.name == "ix_conversation_messages_conversation_status"
        for index in indexes
    )


def test_document_deletion_preflight_checks_scoped_conversations_and_commits_read() -> None:
    session = MagicMock(spec=Session)
    session.scalar.return_value = CONVERSATION_ID
    repository = DocumentRepository(session)

    assert repository.has_scoped_conversation(CONVERSATION_ID) is True
    statement = session.scalar.call_args.args[0]
    sql, parameters = compile_statement(statement)

    assert "conversations.document_id =" in sql
    assert CONVERSATION_ID in parameters.values()
    session.commit.assert_called_once()


def test_conversation_scope_validation_checks_workspace_and_version_parent() -> None:
    document_repository = MagicMock()
    document_repository.get_for_user.return_value = SimpleNamespace(workspace_id=WORKSPACE_ID)
    document_repository.user_can_access_version.return_value = False
    repository = ConversationRepository(MagicMock(), document_repository)

    assert repository.validate_scope(
        WORKSPACE_ID,
        USER_ID,
        UUID("8f3bcfc5-7c76-4196-8dad-214df37f2121"),
        UUID("fe4cc1dc-120e-4575-b0f6-439d0d61d174"),
    ) is False
    document_repository.user_can_access_version.assert_called_once()

    document_repository.get_for_user.return_value = SimpleNamespace(
        workspace_id=UUID("22119a02-61eb-4430-94af-1f8ec1078c7b")
    )
    assert repository.validate_scope(
        WORKSPACE_ID,
        USER_ID,
        UUID("8f3bcfc5-7c76-4196-8dad-214df37f2121"),
        None,
    ) is False
    document_repository.user_can_access_version.assert_called_once()


def test_conversation_history_configuration_defaults_and_invariants() -> None:
    config = Settings(_env_file=None)

    assert config.conversation_max_history_turns == 8
    assert config.conversation_max_history_characters == 12_000
    assert config.conversation_max_question_length == 8_000
    assert config.conversation_stale_pending_seconds == 300
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            conversation_max_history_characters=1_000,
            conversation_max_question_length=2_000,
        )
