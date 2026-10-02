from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.routes.conversations import get_conversation_service
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.conversations.models import ConversationMessageRole, ConversationMessageStatus
from app.conversations.repository import ActiveConversationTurn
from app.conversations.schemas import (
    ConversationCreateRequest,
    ConversationMessageRequest,
)
from app.conversations.service import ConversationService
from app.core.config import settings
from app.core.errors import AppError
from app.main import app
from app.rag.schemas import (
    ConversationContextMessage,
    GroundedAnswerResponse,
    GroundedAnswerRequest,
    GroundedCitation,
)
from app.rag.service import GroundedAnswerService
from app.search.schemas import SemanticSearchResult

USER_ID = UUID("6aa5a694-34c4-44f9-a631-4d9e11f27d1e")
OTHER_USER_ID = UUID("aca07028-48ec-4177-9a5b-6710e11df2ce")
WORKSPACE_ID = UUID("17d67f65-81f3-49db-a199-6357cd8f46f9")
OTHER_WORKSPACE_ID = UUID("cd5b0042-98da-448d-b9ae-55d4558d0b8a")
DOCUMENT_ID = UUID("8f3bcfc5-7c76-4196-8dad-214df37f2121")
VERSION_ID = UUID("fe4cc1dc-120e-4575-b0f6-439d0d61d174")


@dataclass
class FakeConversation:
    id: UUID
    workspace_id: UUID
    owner_user_id: UUID
    title: str | None = None
    document_id: UUID | None = None
    version_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class FakeMessage:
    id: UUID
    conversation_id: UUID
    sequence: int
    role: str
    content: str
    status: str
    citations: list[dict[str, object]] = field(default_factory=list)
    insufficient_context: bool | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class FakeConversationRepository:
    def __init__(self) -> None:
        self.members = {
            (WORKSPACE_ID, USER_ID),
            (WORKSPACE_ID, OTHER_USER_ID),
            (OTHER_WORKSPACE_ID, USER_ID),
        }
        self.scope_is_valid = True
        self.conversations: dict[UUID, FakeConversation] = {}
        self.messages: dict[UUID, list[FakeMessage]] = {}
        self.fail_begin = False
        self.rollback_count = 0

    def is_workspace_member(self, workspace_id: UUID, user_id: UUID) -> bool:
        return (workspace_id, user_id) in self.members

    def validate_scope(
        self,
        workspace_id: UUID,
        user_id: UUID,
        document_id: UUID | None,
        version_id: UUID | None,
    ) -> bool:
        return self.scope_is_valid and (version_id is None or document_id is not None)

    def create(
        self,
        workspace_id: UUID,
        owner_user_id: UUID,
        *,
        title: str | None,
        document_id: UUID | None,
        version_id: UUID | None,
    ) -> FakeConversation:
        now = datetime.now(UTC)
        conversation = FakeConversation(
            id=uuid4(),
            workspace_id=workspace_id,
            owner_user_id=owner_user_id,
            title=title,
            document_id=document_id,
            version_id=version_id,
            created_at=now,
            updated_at=now,
        )
        self.conversations[conversation.id] = conversation
        self.messages[conversation.id] = []
        return conversation

    def list_for_owner(
        self,
        workspace_id: UUID,
        owner_user_id: UUID,
        *,
        limit: int,
        before_updated_at: datetime | None,
        before_id: UUID | None,
    ) -> list[FakeConversation]:
        items = [
            conversation
            for conversation in self.conversations.values()
            if conversation.workspace_id == workspace_id
            and conversation.owner_user_id == owner_user_id
            and self.is_workspace_member(workspace_id, owner_user_id)
        ]
        items.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
        if before_updated_at is not None and before_id is not None:
            items = [
                item
                for item in items
                if (item.updated_at, item.id) < (before_updated_at, before_id)
            ]
        return items[: limit + 1]

    def get_for_owner(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        *,
        for_update: bool = False,
    ) -> FakeConversation | None:
        conversation = self.conversations.get(conversation_id)
        if (
            conversation is None
            or conversation.workspace_id != workspace_id
            or conversation.owner_user_id != owner_user_id
            or not self.is_workspace_member(workspace_id, owner_user_id)
        ):
            return None
        return conversation

    def messages_for_owner(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[FakeMessage]:
        if self.get_for_owner(workspace_id, conversation_id, owner_user_id) is None:
            return []
        items = self.messages[conversation_id]
        if before_sequence is not None:
            items = [item for item in items if item.sequence < before_sequence]
        return items[-(limit + 1) :]

    def begin_user_turn(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        question: str,
        *,
        max_history_turns: int,
        stale_pending_seconds: int,
    ) -> tuple[FakeConversation, FakeMessage, list[FakeMessage]] | None:
        if self.fail_begin:
            raise SQLAlchemyError("database write failed")
        conversation = self.get_for_owner(workspace_id, conversation_id, owner_user_id)
        if conversation is None:
            return None
        messages = self.messages[conversation_id]
        pending = next(
            (
                item
                for item in reversed(messages)
                if item.role == ConversationMessageRole.USER.value
                and item.status == ConversationMessageStatus.PENDING.value
            ),
            None,
        )
        now = datetime.now(UTC)
        if pending is not None:
            if now - pending.created_at < timedelta(seconds=stale_pending_seconds):
                raise ActiveConversationTurn
            pending.status = ConversationMessageStatus.FAILED.value
        history = [
            item for item in messages if item.status == ConversationMessageStatus.COMPLETE.value
        ][-max_history_turns * 2 :]
        next_sequence = max((item.sequence for item in messages), default=0) + 1
        user_message = FakeMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            sequence=next_sequence,
            role=ConversationMessageRole.USER.value,
            content=question,
            status=ConversationMessageStatus.PENDING.value,
            created_at=now,
        )
        messages.append(user_message)
        conversation.updated_at = now
        return conversation, user_message, history

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
    ) -> tuple[FakeMessage, FakeMessage] | None:
        conversation = self.get_for_owner(workspace_id, conversation_id, owner_user_id)
        if conversation is None:
            return None
        messages = self.messages[conversation_id]
        user_message = next((item for item in messages if item.id == user_message_id), None)
        if user_message is None or user_message.status != ConversationMessageStatus.PENDING.value:
            raise RuntimeError("turn is no longer pending")
        user_message.status = ConversationMessageStatus.COMPLETE.value
        assistant_message = FakeMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            sequence=max(item.sequence for item in messages) + 1,
            role=ConversationMessageRole.ASSISTANT.value,
            content=answer,
            status=ConversationMessageStatus.COMPLETE.value,
            citations=citations,
            insufficient_context=insufficient_context,
        )
        messages.append(assistant_message)
        conversation.updated_at = datetime.now(UTC)
        return user_message, assistant_message

    def mark_user_turn_failed(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
        user_message_id: UUID,
    ) -> None:
        messages = self.messages.get(conversation_id, [])
        message = next((item for item in messages if item.id == user_message_id), None)
        if message is not None and message.status == ConversationMessageStatus.PENDING.value:
            message.status = ConversationMessageStatus.FAILED.value

    def delete_for_owner(
        self,
        workspace_id: UUID,
        conversation_id: UUID,
        owner_user_id: UUID,
    ) -> bool:
        if self.get_for_owner(workspace_id, conversation_id, owner_user_id) is None:
            return False
        del self.conversations[conversation_id]
        del self.messages[conversation_id]
        return True

    def rollback(self) -> None:
        self.rollback_count += 1


class FakeGroundedAnswerService:
    def __init__(self, response: GroundedAnswerResponse | None = None) -> None:
        self.response = response or GroundedAnswerResponse(
            answer="The document says it was 12 million dollars [S1].",
            insufficient_context=False,
            citations=[
                GroundedCitation(
                    source_id="S1",
                    document_id=DOCUMENT_ID,
                    filename="report.pdf",
                    version_id=VERSION_ID,
                    version_number=1,
                    chunk_id=uuid4(),
                    chunk_index=0,
                    page_numbers=[2],
                    source_metadata={"page_numbers": [2]},
                )
            ],
        )
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def answer_with_context(
        self,
        workspace_id: UUID,
        user_id: str,
        request: Any,
        *,
        conversation_history,
        retrieval_query: str,
    ) -> GroundedAnswerResponse:
        self.calls.append(
            {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "request": request,
                "history": conversation_history,
                "retrieval_query": retrieval_query,
            }
        )
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def conversation_fixture():
    repository = FakeConversationRepository()
    grounded = FakeGroundedAnswerService()
    service = ConversationService(repository, grounded)  # type: ignore[arg-type]
    return service, repository, grounded


def test_create_conversation_with_fixed_document_version_scope(conversation_fixture) -> None:
    service, repository, _ = conversation_fixture

    response = service.create(
        WORKSPACE_ID,
        str(USER_ID),
        ConversationCreateRequest(
            title=" Quarterly report ",
            document_id=DOCUMENT_ID,
            version_id=VERSION_ID,
        ),
    )

    assert response.title == "Quarterly report"
    assert response.document_id == DOCUMENT_ID
    assert response.version_id == VERSION_ID
    assert response.id in repository.conversations


def test_create_rejects_out_of_workspace_document_scope(conversation_fixture) -> None:
    service, repository, _ = conversation_fixture
    repository.scope_is_valid = False

    with pytest.raises(AppError) as error:
        service.create(
            WORKSPACE_ID,
            str(USER_ID),
            ConversationCreateRequest(document_id=DOCUMENT_ID),
        )

    assert error.value.status_code == 404
    assert repository.conversations == {}


def test_nonmember_cannot_create_or_list_conversations(conversation_fixture) -> None:
    service, repository, _ = conversation_fixture
    repository.members.clear()

    with pytest.raises(AppError) as create_error:
        service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())
    with pytest.raises(AppError) as list_error:
        service.list_conversations(WORKSPACE_ID, str(USER_ID), limit=20)

    assert create_error.value.status_code == list_error.value.status_code == 404


def test_listing_is_owner_and_workspace_scoped(conversation_fixture) -> None:
    service, repository, _ = conversation_fixture
    own = service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())
    service.create(WORKSPACE_ID, str(OTHER_USER_ID), ConversationCreateRequest())
    service.create(OTHER_WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())

    page = service.list_conversations(WORKSPACE_ID, str(USER_ID), limit=20)

    assert [item.id for item in page.items] == [own.id]


def test_detail_delete_and_cross_owner_or_workspace_access_are_private(conversation_fixture) -> None:
    service, repository, _ = conversation_fixture
    created = service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())
    repository.messages[created.id].append(
        FakeMessage(
            uuid4(), created.id, 1, "user", "question", "complete", created_at=datetime.now(UTC)
        )
    )

    detail = service.get(WORKSPACE_ID, created.id, str(USER_ID), message_limit=20)
    assert len(detail.messages) == 1
    for workspace_id, owner_id in [
        (WORKSPACE_ID, OTHER_USER_ID),
        (OTHER_WORKSPACE_ID, USER_ID),
    ]:
        with pytest.raises(AppError) as error:
            service.get(workspace_id, created.id, str(owner_id), message_limit=20)
        assert error.value.status_code == 404
    with pytest.raises(AppError) as delete_error:
        service.delete(WORKSPACE_ID, created.id, str(OTHER_USER_ID))
    assert delete_error.value.status_code == 404

    service.delete(WORKSPACE_ID, created.id, str(USER_ID))
    assert created.id not in repository.conversations
    assert created.id not in repository.messages


def test_missing_conversation_returns_safe_not_found(conversation_fixture) -> None:
    service, _, _ = conversation_fixture

    with pytest.raises(AppError) as error:
        service.get(WORKSPACE_ID, uuid4(), str(USER_ID), message_limit=20)

    assert error.value.status_code == 404


def test_message_persistence_followup_context_order_and_citation_snapshot(
    conversation_fixture,
) -> None:
    service, repository, grounded = conversation_fixture
    conversation = service.create(
        WORKSPACE_ID,
        str(USER_ID),
        ConversationCreateRequest(document_id=DOCUMENT_ID, version_id=VERSION_ID),
    )
    messages = repository.messages[conversation.id]
    messages.extend(
        [
            FakeMessage(uuid4(), conversation.id, 1, "user", "Tell me about revenue", "complete"),
            FakeMessage(uuid4(), conversation.id, 2, "assistant", "Revenue was 12m [S1]", "complete"),
            FakeMessage(
                uuid4(),
                conversation.id,
                3,
                "user",
                "stale pending question",
                "pending",
                created_at=datetime.now(UTC)
                - timedelta(seconds=settings.conversation_stale_pending_seconds + 1),
            ),
            FakeMessage(uuid4(), conversation.id, 4, "user", "failed question", "failed"),
        ]
    )

    response = service.send_message(
        WORKSPACE_ID,
        conversation.id,
        str(USER_ID),
        ConversationMessageRequest(question="What about its margin?", top_k=3),
    )

    assert response.user_message.sequence == 5
    assert response.user_message.status is ConversationMessageStatus.COMPLETE
    assert response.assistant_message.sequence == 6
    assert response.assistant_message.status is ConversationMessageStatus.COMPLETE
    assert response.assistant_message.citations[0].document_id == DOCUMENT_ID
    assert response.assistant_message.citations[0].page_numbers == [2]
    assert [message.role for message in grounded.calls[0]["history"]] == ["user", "assistant"]
    assert grounded.calls[0]["history"][0].content == "Tell me about revenue"
    assert "[S1]" not in grounded.calls[0]["history"][1].content
    assert "Tell me about revenue" in grounded.calls[0]["retrieval_query"]
    assert grounded.calls[0]["request"].question == "What about its margin?"
    assert grounded.calls[0]["request"].document_id == DOCUMENT_ID
    assert grounded.calls[0]["request"].version_id == VERSION_ID


def test_no_retrieved_answer_persists_insufficient_context_without_citations(
    conversation_fixture,
) -> None:
    service, repository, grounded = conversation_fixture
    grounded.response = GroundedAnswerResponse(
        answer="The provided documents do not contain enough information to answer this question.",
        insufficient_context=True,
        citations=[],
    )
    conversation = service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())

    response = service.send_message(
        WORKSPACE_ID,
        conversation.id,
        str(USER_ID),
        ConversationMessageRequest(question="Unknown fact?"),
    )

    assert response.assistant_message.insufficient_context is True
    assert response.assistant_message.citations == []
    assert [item.status for item in repository.messages[conversation.id]] == ["complete", "complete"]


def test_retrieval_failure_retains_failed_user_message_without_assistant(conversation_fixture) -> None:
    service, repository, grounded = conversation_fixture
    grounded.error = AppError(503, "persistence_unavailable", "Search is unavailable.")
    conversation = service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())

    with pytest.raises(AppError) as error:
        service.send_message(
            WORKSPACE_ID,
            conversation.id,
            str(USER_ID),
            ConversationMessageRequest(question="A question"),
        )

    assert error.value.code == "persistence_unavailable"
    assert len(repository.messages[conversation.id]) == 1
    assert repository.messages[conversation.id][0].status == "failed"


def test_generation_failure_is_sanitized_and_retains_only_user_message(conversation_fixture) -> None:
    service, repository, grounded = conversation_fixture
    grounded.error = RuntimeError("secret prompt and API key")
    conversation = service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())

    with pytest.raises(AppError) as error:
        service.send_message(
            WORKSPACE_ID,
            conversation.id,
            str(USER_ID),
            ConversationMessageRequest(question="A question"),
        )

    assert error.value.code == "answer_unavailable"
    assert "secret" not in error.value.message
    assert [item.status for item in repository.messages[conversation.id]] == ["failed"]


def test_initial_message_persistence_failure_does_not_call_rag(conversation_fixture) -> None:
    service, repository, grounded = conversation_fixture
    conversation = service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())
    repository.fail_begin = True

    with pytest.raises(AppError) as error:
        service.send_message(
            WORKSPACE_ID,
            conversation.id,
            str(USER_ID),
            ConversationMessageRequest(question="Question"),
        )

    assert error.value.code == "persistence_unavailable"
    assert grounded.calls == []
    assert repository.rollback_count == 1


def test_active_turn_returns_conflict_and_stale_pending_turn_is_failed(conversation_fixture) -> None:
    service, repository, grounded = conversation_fixture
    conversation = service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())
    pending = FakeMessage(
        uuid4(),
        conversation.id,
        1,
        "user",
        "in progress",
        "pending",
        created_at=datetime.now(UTC),
    )
    repository.messages[conversation.id].append(pending)

    with pytest.raises(AppError) as busy:
        service.send_message(
            WORKSPACE_ID,
            conversation.id,
            str(USER_ID),
            ConversationMessageRequest(question="Another question"),
        )
    assert busy.value.status_code == 409
    assert grounded.calls == []

    pending.created_at = datetime.now(UTC) - timedelta(
        seconds=settings.conversation_stale_pending_seconds + 1
    )
    response = service.send_message(
        WORKSPACE_ID,
        conversation.id,
        str(USER_ID),
        ConversationMessageRequest(question="Retry after stale turn"),
    )
    assert pending.status == "failed"
    assert response.user_message.sequence == 2


def test_bounded_history_keeps_recent_complete_pairs_within_character_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.conversations.service import ConversationService

    monkeypatch.setattr(settings, "conversation_max_history_turns", 3)
    monkeypatch.setattr(settings, "conversation_max_history_characters", 32)
    records = [
        FakeMessage(uuid4(), uuid4(), 1, "user", "old question", "complete"),
        FakeMessage(uuid4(), uuid4(), 2, "assistant", "old answer", "complete"),
        FakeMessage(uuid4(), uuid4(), 3, "user", "recent question", "complete"),
        FakeMessage(uuid4(), uuid4(), 4, "assistant", "recent answer", "complete"),
    ]

    history = ConversationService._bounded_history(records)

    assert [(item.role, item.content) for item in history] == [
        ("user", "recent question"),
        ("assistant", "recent answer"),
    ]
    assert sum(len(item.content) for item in history) <= 32


def test_bounded_history_obeys_recent_turn_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "conversation_max_history_turns", 1)
    monkeypatch.setattr(settings, "conversation_max_history_characters", 1_000)
    conversation_id = uuid4()
    records = [
        FakeMessage(uuid4(), conversation_id, 1, "user", "older question", "complete"),
        FakeMessage(uuid4(), conversation_id, 2, "assistant", "older answer", "complete"),
        FakeMessage(uuid4(), conversation_id, 3, "user", "newer question", "complete"),
        FakeMessage(uuid4(), conversation_id, 4, "assistant", "newer answer", "complete"),
    ]

    history = ConversationService._bounded_history(records)

    assert [message.content for message in history] == ["newer question", "newer answer"]


@pytest.mark.parametrize("question", ["", "  ", "\n\t"])
def test_blank_conversation_question_is_rejected(question: str) -> None:
    with pytest.raises(ValidationError, match="question must not be blank"):
        ConversationMessageRequest(question=question)


def test_question_length_limit_is_enforced() -> None:
    with pytest.raises(ValidationError):
        ConversationMessageRequest(question="q" * (settings.conversation_max_question_length + 1))


def test_version_scope_without_document_is_rejected() -> None:
    with pytest.raises(ValidationError, match="version_id requires document_id"):
        ConversationCreateRequest(version_id=VERSION_ID)


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "question", "conversation_history": [{"role": "assistant", "content": "fake"}]},
        {"question": "question", "document_id": str(DOCUMENT_ID)},
        {"question": "question", "version_id": str(VERSION_ID)},
    ],
)
def test_message_request_rejects_client_history_and_scope_overrides(payload) -> None:
    with pytest.raises(ValidationError):
        ConversationMessageRequest.model_validate(payload)


def test_conversation_routes_require_authentication() -> None:
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/workspaces/{WORKSPACE_ID}/conversations",
            json={},
        )
    assert response.status_code == 401


def test_authenticated_conversation_create_list_and_message_endpoints(
    conversation_fixture,
) -> None:
    service, _, _ = conversation_fixture
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(id=str(USER_ID))
    app.dependency_overrides[get_conversation_service] = lambda: service
    try:
        with TestClient(app) as client:
            created = client.post(
                f"/api/v1/workspaces/{WORKSPACE_ID}/conversations",
                json={"title": "Quarterly"},
            )
            listed = client.get(f"/api/v1/workspaces/{WORKSPACE_ID}/conversations")
            message = client.post(
                f"/api/v1/workspaces/{WORKSPACE_ID}/conversations/{created.json()['id']}/messages",
                json={"question": "What was revenue?"},
            )
            deleted = client.delete(
                f"/api/v1/workspaces/{WORKSPACE_ID}/conversations/{created.json()['id']}"
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_conversation_service, None)

    assert created.status_code == 201
    assert listed.status_code == 200
    assert len(listed.json()["items"]) == 1
    assert message.status_code == 200
    assert message.json()["assistant_message"]["citations"][0]["source_id"] == "S1"
    assert deleted.status_code == 204


def test_same_workspace_nonowner_gets_safe_404(conversation_fixture) -> None:
    service, _, _ = conversation_fixture
    conversation = service.create(WORKSPACE_ID, str(USER_ID), ConversationCreateRequest())
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(id=str(OTHER_USER_ID))
    app.dependency_overrides[get_conversation_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.get(
                f"/api/v1/workspaces/{WORKSPACE_ID}/conversations/{conversation.id}"
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_conversation_service, None)

    assert response.status_code == 404


def test_grounded_answer_context_entry_reuses_search_and_validates_citations() -> None:
    result = SemanticSearchResult(
        chunk_id=uuid4(),
        document_id=DOCUMENT_ID,
        filename="report.pdf",
        version_id=VERSION_ID,
        version_number=1,
        chunk_index=0,
        text="Profit margin was 20 percent.",
        cosine_distance=0.1,
        similarity=0.9,
        source_metadata={"page_numbers": [4]},
    )

    class SearchStub:
        request = None

        def search(self, workspace_id: UUID, user_id: str, request):
            self.request = request
            return [result]

    class GeneratorStub:
        history = None
        question = None

        def generate(self, question: str, sources, history=()):
            self.question = question
            self.history = history
            return type(
                "Generated",
                (),
                {
                    "answer": "Margin was 20 percent [S1], fabricated claim [S8].",
                    "insufficient_context": False,
                },
            )()

    search = SearchStub()
    generator = GeneratorStub()
    grounded = GroundedAnswerService(search, generator)  # type: ignore[arg-type]
    prior = [ConversationContextMessage(role="user", content="Summarize margins")]
    response = grounded.answer_with_context(
        WORKSPACE_ID,
        str(USER_ID),
        GroundedAnswerRequest(
            question="What about this year?",
            document_id=DOCUMENT_ID,
        ),
        conversation_history=prior,
        retrieval_query="Recent context: margins\nWhat about this year?",
    )

    assert search.request.query == "Recent context: margins\nWhat about this year?"
    assert generator.question == "What about this year?"
    assert generator.history == prior
    assert "S8" not in response.answer
    assert [citation.source_id for citation in response.citations] == ["S1"]
    assert response.citations[0].page_numbers == [4]