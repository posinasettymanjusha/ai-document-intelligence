from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Base, DocumentChunkRecord
from app.documents.repository import DocumentRepository
from app.embeddings.models import EmbeddingResult, EmbeddingStatus
from app.embeddings.providers import (
    DeterministicHashEmbeddingProvider,
    EmbeddingProvider,
)
from app.embeddings.service import EmbeddingService


def test_local_provider_implements_interface_and_is_deterministic() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=8)

    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_id == "local-hash-v1"
    assert provider.embed_text("Same text") == provider.embed_text("Same text")
    assert provider.embed_text("Same text") != provider.embed_text("different text")
    assert len(provider.embed_text("Same text")) == 8


def test_provider_batch_matches_single_text_calls() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=8)
    texts = ["one", "two", "three"]

    assert provider.embed_texts(texts) == [provider.embed_text(text) for text in texts]


def test_embedding_service_returns_model_identifier_and_dimensions() -> None:
    provider = DeterministicHashEmbeddingProvider(dimension=8)
    service = EmbeddingService(provider, expected_dimension=8)

    result = service.embed_text("sample content")

    assert result.model_id == provider.model_id
    assert result.dimension == 8
    assert pytest.approx(sum(value * value for value in result.vector)) == 1.0


def test_embedding_service_batch_preserves_input_order() -> None:
    service = EmbeddingService(DeterministicHashEmbeddingProvider(dimension=8), 8)

    results = service.embed_texts(["first", "second"])

    assert len(results) == 2
    assert [result.vector for result in results] == [
        service.embed_text("first").vector,
        service.embed_text("second").vector,
    ]


def test_service_rejects_provider_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match configured dimension"):
        EmbeddingService(DeterministicHashEmbeddingProvider(dimension=8), expected_dimension=9)


def test_service_rejects_invalid_output_dimension_and_nonfinite_values() -> None:
    class IncorrectProvider:
        model_id = "incorrect-test-provider"
        dimension = 3

        def embed_text(self, text: str) -> list[float]:
            return [1.0, 2.0]

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[float("nan"), 0.0, 1.0] for _ in texts]

    service = EmbeddingService(IncorrectProvider(), expected_dimension=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expected 3"):
        service.embed_text("bad dimension")
    with pytest.raises(ValueError, match="finite"):
        service.embed_texts(["nonfinite"])


def test_embedding_service_rejects_provider_batch_count_mismatch() -> None:
    class IncorrectBatchProvider:
        model_id = "incorrect-batch-provider"
        dimension = 2

        def embed_text(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    service = EmbeddingService(IncorrectBatchProvider(), expected_dimension=2)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="different number"):
        service.embed_texts(["one", "two"])


def make_chunk() -> DocumentChunkRecord:
    return DocumentChunkRecord(
        id=uuid4(),
        document_version_id=uuid4(),
        chunk_index=0,
        text="chunk text",
        character_count=10,
        estimated_token_count=3,
        source_metadata={},
        embedding_status=EmbeddingStatus.PROCESSING.value,
        embedding_model="test-model",
    )


def test_repository_stores_and_retrieves_embedding() -> None:
    chunk = make_chunk()
    session = MagicMock(spec=Session)
    session.get.return_value = chunk
    repository = DocumentRepository(session)
    result = EmbeddingResult("test-model", [1.0] * settings.embedding_dimension)

    stored = repository.store_embedding(chunk.id, result)
    retrieved = repository.get_chunk_embedding(chunk.id)

    assert stored.embedding == result.vector
    assert stored.embedding_model == "test-model"
    assert stored.embedding_dimension == settings.embedding_dimension
    assert stored.embedding_status == EmbeddingStatus.READY.value
    assert isinstance(stored.embedding_generated_at, datetime)
    assert retrieved == result
    session.commit.assert_called_once()


def test_repository_stores_batch_embeddings_in_one_commit() -> None:
    chunks = [make_chunk(), make_chunk()]
    session = MagicMock(spec=Session)
    session.scalars.return_value = iter(chunks)
    repository = DocumentRepository(session)
    results = [
        (chunk.id, EmbeddingResult("test-model", [float(index)] * settings.embedding_dimension))
        for index, chunk in enumerate(chunks, start=1)
    ]

    stored = repository.store_embeddings(results)

    assert len(stored) == 2
    assert all(chunk.embedding_status == EmbeddingStatus.READY.value for chunk in stored)
    assert [chunk.embedding[0] for chunk in stored] == [1.0, 2.0]
    session.commit.assert_called_once()


def test_repository_claims_only_pending_chunks_and_marks_them_processing() -> None:
    chunk = make_chunk()
    chunk.embedding_status = EmbeddingStatus.PENDING.value
    chunk.embedding_model = None
    session = MagicMock(spec=Session)
    session.scalars.return_value = iter([chunk])
    repository = DocumentRepository(session)

    claimed = repository.claim_embedding_batch(
        chunk.document_version_id,
        "gemini-embedding-2",
        8,
        retry_failed=False,
    )

    assert claimed == [chunk]
    assert chunk.embedding_status == EmbeddingStatus.PROCESSING.value
    assert chunk.embedding_model == "gemini-embedding-2"
    session.commit.assert_called_once()


def test_repository_failure_update_keeps_ready_vectors_untouched() -> None:
    processing = make_chunk()
    ready = make_chunk()
    ready.embedding_status = EmbeddingStatus.READY.value
    ready.embedding = [1.0] * settings.embedding_dimension
    ready.embedding_dimension = settings.embedding_dimension
    ready.embedding_model = "existing-model"
    ready.embedding_generated_at = datetime.now().astimezone()
    session = MagicMock(spec=Session)
    session.scalars.return_value = iter([processing, ready])
    repository = DocumentRepository(session)

    result = repository.mark_embedding_failures(
        [processing.id, ready.id], "provider unavailable"
    )

    assert [chunk.embedding_status for chunk in result] == [
        EmbeddingStatus.FAILED.value,
        EmbeddingStatus.READY.value,
    ]
    assert processing.embedding_error == "provider unavailable"
    assert ready.embedding == [1.0] * settings.embedding_dimension
    assert ready.embedding_error is None
    session.commit.assert_called_once()


def test_repository_rejects_wrong_dimension_without_committing() -> None:
    chunk = make_chunk()
    session = MagicMock(spec=Session)
    session.get.return_value = chunk
    repository = DocumentRepository(session)

    with pytest.raises(ValueError, match="does not match configured dimension"):
        repository.store_embedding(chunk.id, EmbeddingResult("wrong-size", [0.1, 0.2]))

    session.commit.assert_not_called()


def test_repository_embedding_status_transitions_and_failure_cleanup() -> None:
    chunk = make_chunk()
    session = MagicMock(spec=Session)
    session.get.return_value = chunk
    session.scalars.return_value = iter([chunk])
    repository = DocumentRepository(session)

    processing = repository.mark_embedding_processing(chunk.id, "test-model")
    assert processing.embedding_status == EmbeddingStatus.PROCESSING.value
    assert processing.embedding_model == "test-model"

    repository.store_embedding(
        chunk.id,
        EmbeddingResult("test-model", [0.25] * settings.embedding_dimension),
    )
    assert chunk.embedding_status == EmbeddingStatus.READY.value

    repository.mark_embedding_processing(chunk.id, "test-model")
    failed = repository.mark_embedding_failure(chunk.id)
    assert failed.embedding_status == EmbeddingStatus.FAILED.value
    assert failed.embedding_error == "Embedding generation failed."
    assert failed.embedding is None
    assert failed.embedding_dimension is None
    assert failed.embedding_generated_at is None
    assert session.commit.call_count == 4


def test_pgvector_column_and_hnsw_index_compile_without_database() -> None:
    table = Base.metadata.tables["document_chunks"]
    table_sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
    index_sql = "\n".join(
        str(CreateIndex(index).compile(dialect=postgresql.dialect()))
        for index in table.indexes
        if index.name == "ix_document_chunks_embedding_hnsw_cosine"
    )

    assert f"VECTOR({settings.embedding_dimension})" in table_sql
    assert "USING hnsw" in index_sql
    assert "vector_cosine_ops" in index_sql


def test_pgvector_bind_processor_accepts_configured_length() -> None:
    vector_type = Vector(settings.embedding_dimension)
    processor = vector_type.bind_processor(postgresql.dialect())

    bound = processor([0.0] * settings.embedding_dimension)

    assert bound.startswith("[")
    assert bound.endswith("]")
