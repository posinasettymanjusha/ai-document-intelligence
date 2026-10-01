import hashlib
import logging
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError

from app.chunking.service import DocumentChunkingService
from app.core.config import settings
from app.core.errors import AppError
from app.db.models import Document
from app.documents.errors import DocumentProcessingError
from app.documents.models import DocumentProcessingStatus
from app.documents.repository import DocumentRepository
from app.documents.schemas import (
    DocumentListItemResponse,
    DocumentResponse,
    DocumentVersionResponse,
)
from app.documents.service import DocumentProcessingService
from app.integrations.storage import ObjectStorage, StorageError

logger = logging.getLogger(__name__)


class DocumentLibraryService:
    def __init__(
        self,
        repository: DocumentRepository,
        storage: ObjectStorage,
        processor: DocumentProcessingService,
        chunker: DocumentChunkingService | None = None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._processor = processor
        self._chunker = chunker or DocumentChunkingService(
            chunk_size=settings.document_chunk_size,
            chunk_overlap=settings.document_chunk_overlap,
            minimum_chunk_size=settings.document_minimum_chunk_size,
        )

    def upload(
        self,
        workspace_id: UUID,
        user_id: str,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> DocumentResponse:
        validated = self._processor.validate(filename, content_type, content)
        user_uuid = self._user_uuid(user_id)
        self._require_workspace_membership(workspace_id, user_uuid)
        checksum = hashlib.sha256(validated.content).hexdigest()

        existing = self._database_call(
            lambda: self._repository.get_by_workspace_filename(workspace_id, validated.filename),
            "The document could not be loaded.",
        )
        if existing is not None:
            duplicate = next(
                (version for version in existing.versions if version.checksum == checksum),
                None,
            )
            if duplicate is not None:
                return self._document_response(existing)
            document_id = existing.id
        else:
            document_id = uuid4()

        version_id = uuid4()
        storage_path = (
            f"{workspace_id}/{document_id}/{version_id}/"
            f"{checksum}{validated.extension}"
        )

        try:
            document, version = self._repository.create_version(
                workspace_id=workspace_id,
                document_id=document_id,
                filename=validated.filename,
                file_type=validated.extension.lstrip("."),
                mime_type=validated.content_type,
                file_size=validated.size_bytes,
                storage_path=storage_path,
                checksum=checksum,
            )
        except ValueError:
            latest = self._repository.get_by_workspace_filename(workspace_id, validated.filename)
            if latest is not None and any(item.checksum == checksum for item in latest.versions):
                return self._document_response(latest)
            raise AppError(409, "version_conflict", "A document version conflict occurred.")
        except SQLAlchemyError as error:
            self._rollback_repository()
            raise AppError(503, "persistence_unavailable", "The document could not be recorded.") from error

        try:
            self._storage.upload(storage_path, validated.content, validated.content_type)
        except StorageError as error:
            self._mark_failed(document.id, version.id)
            raise AppError(
                503,
                "storage_unavailable",
                "The uploaded file could not be stored.",
                context={"status": DocumentProcessingStatus.FAILED.value},
            ) from error

        try:
            self._repository.set_status(
                document.id, version.id, DocumentProcessingStatus.PROCESSING
            )
            processed = self._processor.extract(validated)
            chunks = self._chunker.chunk(processed.segments)
            document, _ = self._repository.set_status(
                document.id,
                version.id,
                DocumentProcessingStatus.READY,
                processed,
                chunks,
            )
        except DocumentProcessingError as error:
            self._mark_failed(document.id, version.id)
            error.context["document_id"] = str(document.id)
            raise
        except SQLAlchemyError as error:
            self._rollback_repository()
            self._cleanup_storage(storage_path)
            self._mark_failed(document.id, version.id)
            raise AppError(
                503,
                "persistence_unavailable",
                "The extracted document could not be saved.",
                context={"status": DocumentProcessingStatus.FAILED.value},
            ) from error

        return self._document_response(document)

    def list_documents(
        self, workspace_id: UUID, user_id: str
    ) -> list[DocumentListItemResponse]:
        self._require_workspace_membership(workspace_id, self._user_uuid(user_id))
        documents = self._database_call(
            lambda: self._repository.list_for_workspace(workspace_id),
            "Documents could not be loaded.",
        )
        return [self._list_item(document) for document in documents]

    def get_document(self, document_id: UUID, user_id: str) -> DocumentResponse:
        user_uuid = self._user_uuid(user_id)
        document = self._database_call(
            lambda: self._repository.get_for_user(document_id, user_uuid),
            "The document could not be loaded.",
        )
        if document is None:
            self._not_found()
        return self._document_response(document)

    def delete_document(self, document_id: UUID, user_id: str) -> None:
        user_uuid = self._user_uuid(user_id)
        document = self._database_call(
            lambda: self._repository.get_for_user(
                document_id, user_uuid, include_deleted=True
            ),
            "The document could not be deleted.",
        )
        if document is None:
            self._not_found()

        try:
            if document.deleted_at is None:
                self._repository.soft_delete(document)
            paths = {version.storage_path for version in document.versions}
            paths.add(document.storage_path)
            for path in paths:
                self._storage.delete(path)
            self._repository.delete(document)
        except StorageError as error:
            raise AppError(
                503,
                "storage_unavailable",
                "The document is marked for deletion but its private file could not be removed.",
            ) from error
        except SQLAlchemyError as error:
            self._rollback_repository()
            raise AppError(
                503,
                "persistence_unavailable",
                "The document deletion could not be completed.",
            ) from error

    def _require_workspace_membership(self, workspace_id: UUID, user_id: UUID) -> None:
        allowed = self._database_call(
            lambda: self._repository.is_workspace_member(workspace_id, user_id),
            "Workspace access could not be verified.",
        )
        if not allowed:
            self._not_found()

    def _database_call(self, operation, message: str):
        try:
            return operation()
        except SQLAlchemyError as error:
            self._rollback_repository()
            raise AppError(503, "persistence_unavailable", message) from error

    def _mark_failed(self, document_id: UUID, version_id: UUID) -> None:
        try:
            self._repository.set_status(document_id, version_id, DocumentProcessingStatus.FAILED)
        except SQLAlchemyError:
            self._rollback_repository()
            logger.exception("Could not persist failed document status", extra={"document_id": str(document_id)})

    def _rollback_repository(self) -> None:
        rollback = getattr(self._repository, "rollback", None)
        if rollback is not None:
            rollback()

    def _cleanup_storage(self, path: str) -> None:
        try:
            self._storage.delete(path)
        except StorageError:
            logger.exception("Could not clean up uploaded object", extra={"storage_path": path})

    @staticmethod
    def _user_uuid(user_id: str) -> UUID:
        try:
            return UUID(user_id)
        except ValueError as error:
            raise AppError(401, "unauthorized", "A valid user identity is required.") from error

    @staticmethod
    def _not_found() -> None:
        raise AppError(404, "not_found", "The requested resource was not found.")

    @classmethod
    def _document_response(cls, document: Document) -> DocumentResponse:
        latest = max(document.versions, key=lambda item: item.version_number, default=None)
        return DocumentResponse(
            id=document.id,
            workspace_id=document.workspace_id,
            filename=document.filename,
            file_type=document.file_type,
            mime_type=document.mime_type,
            file_size=document.file_size,
            processing_status=document.processing_status,
            current_version=latest.version_number if latest else None,
            created_at=document.created_at,
            updated_at=document.updated_at,
            versions=[DocumentVersionResponse.model_validate(version) for version in document.versions],
        )

    @staticmethod
    def _list_item(document: Document) -> DocumentListItemResponse:
        latest = max(document.versions, key=lambda item: item.version_number, default=None)
        return DocumentListItemResponse(
            id=document.id,
            workspace_id=document.workspace_id,
            filename=document.filename,
            file_type=document.file_type,
            mime_type=document.mime_type,
            file_size=document.file_size,
            processing_status=document.processing_status,
            current_version=latest.version_number if latest else None,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )