from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes.documents import get_embedding_generation_service
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.core.errors import AppError
from app.embeddings.generation import EmbeddingGenerationService
from app.embeddings.models import EmbeddingResult, EmbeddingStatus
from app.embeddings.service import EmbeddingService
from app.main import app

USER_ID = UUID("7a234e12-45cf-4608-9926-e0dab5c63a94")
DOCUMENT_ID = UUID("8f3bcfc5-7c76-4196-8dad-214df37f2121")
VERSION_ID = UUID("fe4cc1dc-120e-4575-b0f6-439d0d61d174")


@dataclass
class FakeChunk:
    id: UUID
    document_version_id: UUID
    chunk_index: int
    text: str
    embedding_status: str = EmbeddingStatus.PENDING.value
    embedding_model: str | None = None
    embedding: list[float] | None = None
    embedding_dimension: int | None = None
    embedding_generated_at: datetime | None = None
    embedding_error: str | None = None


class FakeRepository:
    def __init__(self, count: int = 3) -> None:
        self.allowed = True
        self.chunks = [
            FakeChunk(uuid4(), VERSION_ID, index, f"chunk {index}")
            for index in range(count)
        ]
        self.fail_store = False
        self.rollback_count = 0

    def user_can_access_version(
        self,
        document_id: UUID,
        version_id: UUID,
        user_id: UUID,
    ) -> bool:
        return self.allowed and document_id == DOCUMENT_ID and version_id == VERSION_ID and user_id == USER_ID

    def claim_embedding_batch(
        self,
        version_id: UUID,
        model_id: str,
        batch_size: int,
        *,
        retry_failed: bool,
    ) -> list[FakeChunk]:
        eligible = {EmbeddingStatus.PENDING.value}
        if retry_failed:
            eligible.add(EmbeddingStatus.FAILED.value)
        selected = [
            chunk
            for chunk in self.chunks
            if chunk.document_version_id == version_id and chunk.embedding_status in eligible
        ][:batch_size]
        for chunk in selected:
            chunk.embedding = None
            chunk.embedding_dimension = None
            chunk.embedding_generated_at = None
            chunk.embedding_error = None
            chunk.embedding_model = model_id
            chunk.embedding_status = EmbeddingStatus.PROCESSING.value
        return selected

    def store_embeddings(self, embeddings: list[tuple[UUID, EmbeddingResult]]) -> None:
        if self.fail_store:
            raise RuntimeError("database commit failed")
        selected = {chunk.id: chunk for chunk in self.chunks}
        for chunk_id, embedding in embeddings:
            chunk = selected[chunk_id]
            if chunk.embedding_status != EmbeddingStatus.PROCESSING.value:
                raise ValueError("not processing")
            if embedding.dimension != 4 or embedding.model_id != chunk.embedding_model:
                raise ValueError("invalid result")
        now = datetime.now(UTC)
        for chunk_id, embedding in embeddings:
            chunk = selected[chunk_id]
            chunk.embedding = embedding.vector
            chunk.embedding_dimension = embedding.dimension
            chunk.embedding_generated_at = now
            chunk.embedding_status = EmbeddingStatus.READY.value
            chunk.embedding_error = None

    def mark_embedding_failures(self, chunk_ids: list[UUID], message: str) -> list[FakeChunk]:
        selected = {chunk.id: chunk for chunk in self.chunks}
        for chunk_id in chunk_ids:
            chunk = selected[chunk_id]
            if chunk.embedding_status == EmbeddingStatus.PROCESSING.value:
                chunk.embedding = None
                chunk.embedding_dimension = None
                chunk.embedding_generated_at = None
                chunk.embedding_error = message[:500]
                chunk.embedding_status = EmbeddingStatus.FAILED.value
        return [selected[chunk_id] for chunk_id in chunk_ids]

    def rollback(self) -> None:
        self.rollback_count += 1


class CountingProvider:
    model_id = "fake-model"
    dimension = 4

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_text(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 2.0, 3.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [self.embed_text(text) for text in texts]


class FailingProvider(CountingProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        raise RuntimeError("request contained secret API key SHOULD_NOT_PERSIST")


@pytest.fixture
def generation() -> tuple[EmbeddingGenerationService, FakeRepository, CountingProvider]:
    repository = FakeRepository()
    provider = CountingProvider()
    service = EmbeddingGenerationService(
        repository,
        EmbeddingService(provider, expected_dimension=4),
        batch_size=2,
    )
    return service, repository, provider


def test_generation_batches_chunks_and_marks_ready(
    generation: tuple[EmbeddingGenerationService, FakeRepository, CountingProvider],
) -> None:
    service, repository, provider = generation

    summary = service.generate_for_version(DOCUMENT_ID, VERSION_ID, str(USER_ID))

    assert summary.attempted_chunks == summary.ready_chunks == 3
    assert summary.failed_chunks == 0
    assert provider.batch_sizes == [2, 1]
    assert all(chunk.embedding_status == EmbeddingStatus.READY.value for chunk in repository.chunks)
    assert all(chunk.embedding_dimension == 4 for chunk in repository.chunks)


def test_ready_chunks_are_not_reclaimed_or_overwritten(
    generation: tuple[EmbeddingGenerationService, FakeRepository, CountingProvider],
) -> None:
    service, repository, _ = generation
    ready = repository.chunks[0]
    ready.embedding_status = EmbeddingStatus.READY.value
    ready.embedding = [9.0, 8.0, 7.0, 6.0]
    ready.embedding_dimension = 4
    ready.embedding_model = "previous-model"

    summary = service.generate_for_version(DOCUMENT_ID, VERSION_ID, str(USER_ID))

    assert summary.attempted_chunks == 2
    assert ready.embedding == [9.0, 8.0, 7.0, 6.0]
    assert ready.embedding_model == "previous-model"


def test_provider_failure_marks_batch_failed_without_persisting_secret() -> None:
    repository = FakeRepository(count=2)
    service = EmbeddingGenerationService(
        repository,
        EmbeddingService(FailingProvider(), expected_dimension=4),
        batch_size=2,
    )

    summary = service.generate_for_version(DOCUMENT_ID, VERSION_ID, str(USER_ID))

    assert summary.failed_chunks == 2
    assert summary.ready_chunks == 0
    assert summary.failure_message == "Embedding provider request failed (RuntimeError)."
    assert all(chunk.embedding_status == EmbeddingStatus.FAILED.value for chunk in repository.chunks)
    assert all("SHOULD_NOT_PERSIST" not in chunk.embedding_error for chunk in repository.chunks)


def test_failed_chunks_can_be_explicitly_retried() -> None:
    repository = FakeRepository(count=2)
    provider = FailingProvider()
    service = EmbeddingGenerationService(
        repository,
        EmbeddingService(provider, expected_dimension=4),
        batch_size=2,
    )
    failed = service.generate_for_version(DOCUMENT_ID, VERSION_ID, str(USER_ID))
    assert failed.failed_chunks == 2

    succeeding_provider = CountingProvider()
    retry_service = EmbeddingGenerationService(
        repository,
        EmbeddingService(succeeding_provider, expected_dimension=4),
        batch_size=2,
    )
    retried = retry_service.generate_for_version(
        DOCUMENT_ID,
        VERSION_ID,
        str(USER_ID),
        retry_failed=True,
    )

    assert retried.ready_chunks == 2
    assert all(chunk.embedding_status == EmbeddingStatus.READY.value for chunk in repository.chunks)
    assert all(chunk.embedding_error is None for chunk in repository.chunks)


def test_batch_persistence_failure_rolls_back_and_marks_claimed_chunks_failed() -> None:
    repository = FakeRepository(count=2)
    repository.fail_store = True
    service = EmbeddingGenerationService(
        repository,
        EmbeddingService(CountingProvider(), expected_dimension=4),
        batch_size=2,
    )

    summary = service.generate_for_version(DOCUMENT_ID, VERSION_ID, str(USER_ID))

    assert summary.ready_chunks == 0
    assert summary.failed_chunks == 2
    assert repository.rollback_count == 1
    assert all(chunk.embedding_status == EmbeddingStatus.FAILED.value for chunk in repository.chunks)
    assert all(chunk.embedding is None for chunk in repository.chunks)


def test_generation_hides_versions_and_chunks_from_other_users() -> None:
    repository = FakeRepository(count=1)
    repository.allowed = False
    provider = CountingProvider()
    service = EmbeddingGenerationService(
        repository,
        EmbeddingService(provider, expected_dimension=4),
        batch_size=1,
    )

    with pytest.raises(AppError) as error:
        service.generate_for_version(DOCUMENT_ID, VERSION_ID, str(USER_ID))

    assert error.value.status_code == 404
    assert provider.batch_sizes == []


def test_explicit_generation_route_uses_auth_and_service_overrides(
    generation: tuple[EmbeddingGenerationService, FakeRepository, CountingProvider],
) -> None:
    service, _, _ = generation
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(id=str(USER_ID))
    app.dependency_overrides[get_embedding_generation_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/documents/{DOCUMENT_ID}/versions/{VERSION_ID}/embeddings",
                json={"retry_failed": False},
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_embedding_generation_service, None)

    assert response.status_code == 200
    assert response.json()["ready_chunks"] == 3
    assert response.json()["failed_chunks"] == 0
