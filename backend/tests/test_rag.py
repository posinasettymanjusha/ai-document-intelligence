from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.v1.routes.answers import get_grounded_answer_service
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.core.config import settings
from app.core.errors import AppError
from app.embeddings.service import EmbeddingService
from app.integrations.gemini_generation import GeminiGenerationProvider
from app.main import app
from app.rag.schemas import GeneratedAnswer, GroundedAnswerRequest
from app.rag.service import (
    AnswerGenerationError,
    AnswerGenerationTimeout,
    GroundedAnswerService,
)
from app.search.schemas import SemanticSearchResult
from app.search.service import SemanticSearchService

USER_ID = UUID("6aa5a694-34c4-44f9-a631-4d9e11f27d1e")
WORKSPACE_ID = UUID("17d67f65-81f3-49db-a199-6357cd8f46f9")
DOCUMENT_ID = UUID("aca07028-48ec-4177-9a5b-6710e11df2ce")
VERSION_ID = UUID("cd5b0042-98da-448d-b9ae-55d4558d0b8a")


def make_result(
    *,
    chunk_id: UUID | None = None,
    chunk_index: int = 2,
    metadata: dict[str, object] | None = None,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        chunk_id=chunk_id or uuid4(),
        document_id=DOCUMENT_ID,
        filename="report.pdf",
        version_id=VERSION_ID,
        version_number=4,
        chunk_index=chunk_index,
        text="Annual revenue was 12 million dollars.",
        cosine_distance=0.05,
        similarity=0.95,
        source_metadata=metadata
        or {"page_numbers": [2], "sources": [{"page_number": 2, "segment_sequence": 2}]},
    )


class FakeSearchService:
    def __init__(self, results: list[SemanticSearchResult] | None = None) -> None:
        self.results = [make_result()] if results is None else results
        self.calls: list[tuple[UUID, str, Any]] = []

    def search(self, workspace_id: UUID, user_id: str, request: Any) -> list[SemanticSearchResult]:
        self.calls.append((workspace_id, user_id, request))
        return self.results


class FakeGenerator:
    def __init__(self, response: GeneratedAnswer | None = None) -> None:
        self.response = response or GeneratedAnswer(
            answer="Annual revenue was 12 million dollars [S1].",
            insufficient_context=False,
        )
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    def generate(self, question: str, sources: list[tuple[str, str]]) -> GeneratedAnswer:
        self.calls.append((question, sources))
        return self.response


@dataclass
class FakeChunk:
    id: UUID
    chunk_index: int
    text: str
    source_metadata: dict[str, object]


@dataclass
class FakeDocument:
    id: UUID
    filename: str


@dataclass
class FakeVersion:
    id: UUID
    version_number: int


class FakeProvider:
    model_id = "test-embedding-model"
    dimension = settings.embedding_dimension

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0] + [0.0] * (self.dimension - 1)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class DeniedRepository:
    def __init__(self) -> None:
        self.search_called = False

    def is_workspace_member(self, workspace_id: UUID, user_id: UUID) -> bool:
        return False

    def search_chunks(self, **kwargs: Any) -> list[Any]:
        self.search_called = True
        return []

    def rollback(self) -> None:
        return None

    def commit(self) -> None:
        return None


def test_successful_retrieval_generates_answer_and_maps_citations() -> None:
    first = make_result()
    second = make_result(
        chunk_index=3,
        metadata={"sources": [{"line_number": 8}], "line_start": 8, "line_end": 8},
    )
    search = FakeSearchService([first, second])
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="Revenue was 12 million dollars [S1], with details in the next section [S2].",
            insufficient_context=False,
        )
    )
    service = GroundedAnswerService(search, generator)  # type: ignore[arg-type]

    response = service.answer(
        WORKSPACE_ID,
        str(USER_ID),
        GroundedAnswerRequest(
            question="  What was revenue?  ",
            top_k=2,
            document_id=DOCUMENT_ID,
            version_id=VERSION_ID,
        ),
    )

    assert response.insufficient_context is False
    assert response.answer.endswith("[S2].")
    assert [citation.source_id for citation in response.citations] == ["S1", "S2"]
    assert response.citations[0].document_id == DOCUMENT_ID
    assert response.citations[0].filename == "report.pdf"
    assert response.citations[0].version_id == VERSION_ID
    assert response.citations[0].version_number == 4
    assert response.citations[0].chunk_id == first.chunk_id
    assert response.citations[0].chunk_index == first.chunk_index
    assert response.citations[0].page_numbers == [2]
    assert response.citations[0].source_metadata == first.source_metadata
    assert response.citations[1].page_numbers == []
    assert response.citations[1].source_metadata == second.source_metadata
    assert search.calls[0][2].query == "What was revenue?"
    assert search.calls[0][2].top_k == 2
    assert search.calls[0][2].document_id == DOCUMENT_ID
    assert search.calls[0][2].version_id == VERSION_ID
    assert generator.calls == [
        (
            "What was revenue?",
            [
                ("S1", first.text),
                ("S2", second.text),
            ],
        )
    ]


def test_no_results_returns_insufficient_context_without_generation() -> None:
    generator = FakeGenerator()
    service = GroundedAnswerService(FakeSearchService([]), generator)  # type: ignore[arg-type]

    response = service.answer(
        WORKSPACE_ID,
        str(USER_ID),
        GroundedAnswerRequest(question="What was revenue?"),
    )

    assert response.insufficient_context is True
    assert "not contain enough information" in response.answer
    assert response.citations == []
    assert generator.calls == []


def test_no_results_does_not_require_or_create_gemini_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.v1.routes import answers as answers_route

    monkeypatch.setattr(settings, "gemini_api_key", None)
    provider = answers_route.get_gemini_generation_provider()
    service = GroundedAnswerService(
        FakeSearchService([]),
        provider,
    )

    response = service.answer(
        WORKSPACE_ID,
        str(USER_ID),
        GroundedAnswerRequest(question="What was revenue?"),
    )

    assert response.insufficient_context is True
    assert isinstance(provider, GeminiGenerationProvider)


def test_generator_insufficient_context_discards_answer_and_citations() -> None:
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="An unsupported guess [S1].",
            insufficient_context=True,
        )
    )
    service = GroundedAnswerService(FakeSearchService(), generator)  # type: ignore[arg-type]

    response = service.answer(
        WORKSPACE_ID,
        str(USER_ID),
        GroundedAnswerRequest(question="What is the answer?"),
    )

    assert response.insufficient_context is True
    assert response.answer == "The provided documents do not contain enough information to answer this question."
    assert response.citations == []


def test_unknown_citation_labels_are_removed_and_not_mapped() -> None:
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="The value is supported [S1]; fabricated source [S99].",
            insufficient_context=False,
        )
    )
    service = GroundedAnswerService(FakeSearchService(), generator)  # type: ignore[arg-type]

    response = service.answer(
        WORKSPACE_ID,
        str(USER_ID),
        GroundedAnswerRequest(question="What is the value?"),
    )

    assert "S99" not in response.answer
    assert [citation.source_id for citation in response.citations] == ["S1"]


def test_answer_without_known_citations_fails_closed() -> None:
    generator = FakeGenerator(
        GeneratedAnswer(answer="An uncited assertion.", insufficient_context=False)
    )
    service = GroundedAnswerService(FakeSearchService(), generator)  # type: ignore[arg-type]

    response = service.answer(
        WORKSPACE_ID,
        str(USER_ID),
        GroundedAnswerRequest(question="What is the answer?"),
    )

    assert response.insufficient_context is True
    assert response.citations == []


def test_workspace_authorization_precedes_embedding_and_generation() -> None:
    repository = DeniedRepository()
    embedding_provider = FakeProvider()
    search_service = SemanticSearchService(
        repository,  # type: ignore[arg-type]
        EmbeddingService(embedding_provider),
    )
    generator = FakeGenerator()
    service = GroundedAnswerService(search_service, generator)  # type: ignore[arg-type]

    with pytest.raises(AppError) as error:
        service.answer(
            WORKSPACE_ID,
            str(USER_ID),
            GroundedAnswerRequest(question="private question"),
        )

    assert error.value.status_code == 404
    assert embedding_provider.calls == []
    assert generator.calls == []
    assert repository.search_called is False


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_is_rejected(question: str) -> None:
    with pytest.raises(ValidationError, match="question must not be blank"):
        GroundedAnswerRequest(question=question)


@pytest.mark.parametrize("top_k", [0, 11])
def test_top_k_outside_rag_bounds_is_rejected(top_k: int) -> None:
    with pytest.raises(ValidationError):
        GroundedAnswerRequest(question="valid", top_k=top_k)


def test_rag_request_defaults_to_five_sources_and_accepts_filters() -> None:
    request = GroundedAnswerRequest(
        question="valid question",
        document_id=DOCUMENT_ID,
        version_id=VERSION_ID,
    )

    assert request.top_k == 5
    assert request.document_id == DOCUMENT_ID
    assert request.version_id == VERSION_ID


def test_endpoint_requires_authentication() -> None:
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/workspaces/{WORKSPACE_ID}/answers",
            json={"question": "annual revenue"},
        )

    assert response.status_code == 401


def test_authenticated_endpoint_returns_grounded_response() -> None:
    generator = FakeGenerator()
    service = GroundedAnswerService(FakeSearchService(), generator)  # type: ignore[arg-type]
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(id=str(USER_ID))
    app.dependency_overrides[get_grounded_answer_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/workspaces/{WORKSPACE_ID}/answers",
                json={"question": "annual revenue"},
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_grounded_answer_service, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].endswith("[S1].")
    assert payload["insufficient_context"] is False
    assert payload["citations"][0]["source_id"] == "S1"
    assert payload["citations"][0]["source_metadata"]["page_numbers"] == [2]


def test_endpoint_rejects_out_of_range_top_k() -> None:
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(id=str(USER_ID))
    app.dependency_overrides[get_grounded_answer_service] = lambda: GroundedAnswerService(
        FakeSearchService(), FakeGenerator()
    )  # type: ignore[arg-type]
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/workspaces/{WORKSPACE_ID}/answers",
                json={"question": "annual revenue", "top_k": 11},
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_grounded_answer_service, None)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status_code", "code", "safe_message"),
    [
        (
            AnswerGenerationTimeout("provider secret"),
            504,
            "answer_generation_timeout",
            "Answer generation timed out. Please try again.",
        ),
        (
            AnswerGenerationError("provider secret"),
            503,
            "answer_generation_unavailable",
            "An answer could not be generated right now.",
        ),
    ],
)
def test_provider_failures_are_sanitized(
    error: Exception,
    status_code: int,
    code: str,
    safe_message: str,
) -> None:
    class FailingGenerator:
        def generate(self, question: str, sources: list[tuple[str, str]]) -> GeneratedAnswer:
            raise error

    service = GroundedAnswerService(FakeSearchService(), FailingGenerator())  # type: ignore[arg-type]

    with pytest.raises(AppError) as raised:
        service.answer(
            WORKSPACE_ID,
            str(USER_ID),
            GroundedAnswerRequest(question="question"),
        )

    assert raised.value.status_code == status_code
    assert raised.value.code == code
    assert raised.value.message == safe_message
    assert "provider secret" not in raised.value.message