from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import AppError
from app.documents.repository import DocumentRepository
from app.embeddings.models import EmbeddingGenerationSummary, EmbeddingStatus
from app.embeddings.service import EmbeddingService


class EmbeddingGenerationService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_service: EmbeddingService,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self._repository = repository
        self._embedding_service = embedding_service
        self._batch_size = batch_size

    def generate_for_version(
        self,
        document_id: UUID,
        version_id: UUID,
        user_id: str,
        *,
        retry_failed: bool = False,
    ) -> EmbeddingGenerationSummary:
        try:
            user_uuid = UUID(user_id)
        except ValueError as error:
            raise AppError(401, "unauthorized", "A valid user identity is required.") from error

        try:
            allowed = self._repository.user_can_access_version(
                document_id, version_id, user_uuid
            )
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(503, "persistence_unavailable", "Document access could not be verified.") from error
        if not allowed:
            raise AppError(404, "not_found", "The requested resource was not found.")

        attempted = 0
        ready = 0
        failed = 0
        failure_message: str | None = None
        while True:
            try:
                batch = self._repository.claim_embedding_batch(
                    version_id,
                    self._embedding_service.model_id,
                    self._batch_size,
                    retry_failed=retry_failed,
                )
            except SQLAlchemyError as error:
                self._repository.rollback()
                raise AppError(503, "persistence_unavailable", "Embedding work could not be claimed.") from error
            if not batch:
                break

            chunk_ids = [chunk.id for chunk in batch]
            attempted += len(batch)
            try:
                results = self._embedding_service.embed_texts([chunk.text for chunk in batch])
            except Exception as error:
                self._repository.rollback()
                failure_message = self._safe_failure_message(error)
                batch_ready, batch_failed = self._mark_batch_failed(chunk_ids, failure_message)
                ready += batch_ready
                failed += batch_failed
                break

            try:
                self._repository.store_embeddings(list(zip(chunk_ids, results, strict=True)))
                ready += len(batch)
            except Exception as error:
                self._repository.rollback()
                failure_message = "Embedding batch could not be persisted; retry the failed chunks."
                batch_ready, batch_failed = self._mark_batch_failed(chunk_ids, failure_message)
                ready += batch_ready
                failed += batch_failed
                break

        return EmbeddingGenerationSummary(
            attempted_chunks=attempted,
            ready_chunks=ready,
            failed_chunks=failed,
            failure_message=failure_message,
        )

    def _mark_batch_failed(self, chunk_ids: list[UUID], message: str) -> tuple[int, int]:
        try:
            chunks = self._repository.mark_embedding_failures(chunk_ids, message)
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "Embedding failed and its failure status could not be saved.",
            ) from error
        ready = sum(chunk.embedding_status == EmbeddingStatus.READY.value for chunk in chunks)
        failed = sum(chunk.embedding_status == EmbeddingStatus.FAILED.value for chunk in chunks)
        return ready, failed

    @staticmethod
    def _safe_failure_message(error: Exception) -> str:
        status_code = getattr(error, "code", None)
        if isinstance(status_code, int):
            return f"Embedding provider request failed with status {status_code}."
        if isinstance(error, ValueError):
            return "Embedding provider returned invalid data."
        return f"Embedding provider request failed ({type(error).__name__})."