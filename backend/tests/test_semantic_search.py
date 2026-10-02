from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.api.v1.routes.search import get_semantic_search_service
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.core.config import settings
from app.core.errors import AppError
from app.documents.repository import DocumentRepository
from app.embeddings.service import EmbeddingService
from app.main import app
from app.search.schemas import SemanticSearchRequest
from app.search.service import SemanticSearchService

USER_ID = UUID("6aa5a694-34c4-44f9-a631-4d9e11f27d1e")
WORKSPACE_ID = UUID("17d67f65-81f3-49db-a199-6357cd8f46f9")
DOCUMENT_ID = UUID("aca07028-48ec-4177-9a5b-6710e11df2ce")
VERSION_ID = UUID("cd5b0042-98da-448d-b9ae-55d4558d0b8a")


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
    model_id = "semantic-test-model"
    dimension = settings.embedding_dimension

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.events: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        self.events.append("embed")
        return [1.0] + [0.0] * (self.dimension - 1)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class FakeRepository:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.search_arguments: dict[str, Any] | None = None
        self.events: list[str] = []
        self.matches = [
            (
                FakeChunk(
                    uuid4(),
                    1,
                    "closest chunk",
                    {"page_numbers": [2], "sources": [{"page_number": 2}]},
                ),
                FakeDocument(DOCUMENT_ID, "report.pdf"),
                FakeVersion(VERSION_ID, 4),
                0.05,
            ),
            (
                FakeChunk(
                    uuid4(),
                    2,
                    "second closest chunk",
                    {"page_numbers": [3], "sources": [{"page_number": 3}]},
                ),
                FakeDocument(DOCUMENT_ID, "report.pdf"),
                FakeVersion(VERSION_ID, 4),
                0.2,
            ),
        ]

    def is_workspace_member(self, workspace_id: UUID, user_id: UUID) -> bool:
        self.events.append("membership")
        return self.allowed and workspace_id == WORKSPACE_ID and user_id == USER_ID

    def search_chunks(self, **kwargs: Any) -> list[tuple[FakeChunk, FakeDocument, FakeVersion, float]]:
        self.events.append("search")
        self.search_arguments = kwargs
        return self.matches

    def rollback(self) -> None:
        return None

    def commit(self) -> None:
        self.events.append("commit")


@dataclass
class SearchFixture:
    service: SemanticSearchService
    repository: FakeRepository
    provider: FakeProvider


@pytest.fixture
def search_fixture() -> SearchFixture:
    repository = FakeRepository()
    provider = FakeProvider()
    repository.events = provider.events
    service = SemanticSearchService(repository, EmbeddingService(provider))  # type: ignore[arg-type]
    return SearchFixture(service, repository, provider)


def test_search_embeds_query_and_returns_ranked_scores_and_source_metadata(
    search_fixture: SearchFixture,
) -> None:
    result = search_fixture.service.search(
        WORKSPACE_ID,
        str(USER_ID),
        SemanticSearchRequest(query="  annual revenue  ", top_k=7),
    )

    assert search_fixture.provider.queries == ["annual revenue"]
    assert search_fixture.repository.search_arguments is not None
    assert search_fixture.repository.search_arguments["query_vector"] == [
        1.0,
        *([0.0] * (settings.embedding_dimension - 1)),
    ]
    assert search_fixture.repository.search_arguments["embedding_model"] == "semantic-test-model"
    assert search_fixture.repository.search_arguments["top_k"] == 7
    assert [item.text for item in result] == ["closest chunk", "second closest chunk"]
    assert [item.cosine_distance for item in result] == [0.05, 0.2]
    assert [item.similarity for item in result] == pytest.approx([0.95, 0.8])
    assert result[0].filename == "report.pdf"
    assert result[0].document_id == DOCUMENT_ID
    assert result[0].version_id == VERSION_ID
    assert result[0].version_number == 4
    assert result[0].chunk_index == 1
    assert result[0].source_metadata["page_numbers"] == [2]
    assert search_fixture.provider.events == [
        "membership",
        "commit",
        "embed",
        "search",
        "commit",
    ]


def test_search_rejects_non_member_before_embedding() -> None:
    repository = FakeRepository(allowed=False)
    provider = FakeProvider()
    service = SemanticSearchService(repository, EmbeddingService(provider))  # type: ignore[arg-type]

    with pytest.raises(AppError) as error:
        service.search(
            WORKSPACE_ID,
            str(USER_ID),
            SemanticSearchRequest(query="private query"),
        )

    assert error.value.status_code == 404
    assert provider.queries == []
    assert repository.search_arguments is None


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(query: str) -> None:
    with pytest.raises(ValidationError, match="query must not be blank"):
        SemanticSearchRequest(query=query)


@pytest.mark.parametrize("top_k", [0, 51])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="valid query", top_k=top_k)


def compile_search_sql(
    *,
    document_id: UUID | None = None,
    version_id: UUID | None = None,
) -> tuple[str, dict[str, Any]]:
    session = MagicMock(spec=Session)
    session.execute.return_value.all.return_value = []
    repository = DocumentRepository(session)
    repository.search_chunks(
        workspace_id=WORKSPACE_ID,
        user_id=USER_ID,
        query_vector=[0.0] * settings.embedding_dimension,
        embedding_model="semantic-test-model",
        top_k=12,
        document_id=document_id,
        version_id=version_id,
    )
    statement = session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    return str(compiled), compiled.params


def test_repository_query_uses_cosine_order_and_security_embedding_filters() -> None:
    sql, parameters = compile_search_sql()

    assert "document_chunks.embedding <=>" in sql
    assert "ORDER BY (document_chunks.embedding <=>" in sql
    assert " ASC" in sql
    assert "LIMIT" in sql
    assert "workspace_members.user_id" in sql
    assert "documents.workspace_id" in sql
    assert "documents.deleted_at IS NULL" in sql
    assert "document_chunks.embedding_status" in sql
    assert "document_chunks.embedding IS NOT NULL" in sql
    assert "document_chunks.embedding_model" in sql
    assert WORKSPACE_ID in parameters.values()
    assert USER_ID in parameters.values()
    assert "ready" in parameters.values()
    assert "semantic-test-model" in parameters.values()


def test_repository_query_applies_document_filter() -> None:
    sql, parameters = compile_search_sql(document_id=DOCUMENT_ID)
    where_clause = sql.split("WHERE", 1)[1].split("ORDER BY", 1)[0]

    assert "documents.id =" in where_clause
    assert DOCUMENT_ID in parameters.values()
    assert "document_versions.id =" not in where_clause


def test_repository_query_applies_version_filter() -> None:
    sql, parameters = compile_search_sql(version_id=VERSION_ID)
    where_clause = sql.split("WHERE", 1)[1].split("ORDER BY", 1)[0]

    assert "document_versions.id =" in where_clause
    assert VERSION_ID in parameters.values()
    assert "documents.id =" not in where_clause


def test_repository_query_combines_document_and_version_filters() -> None:
    sql, parameters = compile_search_sql(document_id=DOCUMENT_ID, version_id=VERSION_ID)
    where_clause = sql.split("WHERE", 1)[1].split("ORDER BY", 1)[0]

    assert "documents.id =" in where_clause
    assert "document_versions.id =" in where_clause
    assert DOCUMENT_ID in parameters.values()
    assert VERSION_ID in parameters.values()


def test_search_endpoint_requires_authentication() -> None:
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/workspaces/{WORKSPACE_ID}/search",
            json={"query": "annual revenue"},
        )

    assert response.status_code == 401


def test_authenticated_search_endpoint_returns_search_results(
    search_fixture: SearchFixture,
) -> None:
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(id=str(USER_ID))
    app.dependency_overrides[get_semantic_search_service] = lambda: search_fixture.service
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/workspaces/{WORKSPACE_ID}/search",
                json={
                    "query": "annual revenue",
                    "top_k": 5,
                    "document_id": str(DOCUMENT_ID),
                    "version_id": str(VERSION_ID),
                },
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_semantic_search_service, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["text"] == "closest chunk"
    assert payload[0]["cosine_distance"] == 0.05
    assert payload[0]["similarity"] == pytest.approx(0.95)
    assert payload[0]["source_metadata"]["page_numbers"] == [2]
    assert search_fixture.repository.search_arguments["document_id"] == DOCUMENT_ID
    assert search_fixture.repository.search_arguments["version_id"] == VERSION_ID