from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.routes.documents import get_document_library_service
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.chunking.models import DocumentChunk
from app.core.errors import AppError
from app.db.models import Document, DocumentVersion
from app.documents.library import DocumentLibraryService
from app.documents.models import DocumentProcessingStatus, ProcessedDocument
from app.documents.service import DocumentProcessingService
from app.integrations.storage import StorageError
from app.main import app

USER_ID = UUID("54ab4cae-344a-4ad8-9476-02c2bef3e574")
WORKSPACE_ID = UUID("46414b79-df09-4585-9cd1-7d3fa54d4adb")
OTHER_WORKSPACE_ID = UUID("22119a02-61eb-4430-94af-1f8ec1078c7b")


class MemoryRepository:
    def __init__(self) -> None:
        self.documents: dict[UUID, Document] = {}
        self.chunks: dict[UUID, list[DocumentChunk]] = {}
        self.members = {(WORKSPACE_ID, USER_ID)}
        self.fail_start = False
        self.scoped_documents: set[UUID] = set()

    def is_workspace_member(self, workspace_id: UUID, user_id: UUID) -> bool:
        return (workspace_id, user_id) in self.members

    def get_by_workspace_filename(self, workspace_id: UUID, filename: str) -> Document | None:
        return next(
            (
                document
                for document in self.documents.values()
                if document.workspace_id == workspace_id
                and document.filename == filename
                and document.deleted_at is None
            ),
            None,
        )

    def get_for_user(
        self,
        document_id: UUID,
        user_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> Document | None:
        document = self.documents.get(document_id)
        if (
            document is None
            or (document.deleted_at is not None and not include_deleted)
            or (document.workspace_id, user_id) not in self.members
        ):
            return None
        return document

    def list_for_workspace(self, workspace_id: UUID) -> list[Document]:
        return [
            document
            for document in self.documents.values()
            if document.workspace_id == workspace_id and document.deleted_at is None
        ]

    def create_version(
        self,
        workspace_id: UUID,
        document_id: UUID,
        filename: str,
        file_type: str,
        mime_type: str,
        file_size: int,
        storage_path: str,
        checksum: str,
    ) -> tuple[Document, DocumentVersion]:
        if self.fail_start:
            raise SQLAlchemyError("database unavailable")
        document = self.documents.get(document_id)
        if document is None:
            now = datetime.now(UTC)
            document = Document(
                id=document_id,
                workspace_id=workspace_id,
                filename=filename,
                file_type=file_type,
                mime_type=mime_type,
                file_size=file_size,
                storage_path=storage_path,
                processing_status=DocumentProcessingStatus.UPLOADED.value,
                created_at=now,
                updated_at=now,
                versions=[],
            )
            self.documents[document_id] = document
        version = DocumentVersion(
            id=uuid4(),
            document_id=document.id,
            version_number=len(document.versions) + 1,
            checksum=checksum,
            original_filename=filename,
            file_type=file_type,
            mime_type=mime_type,
            file_size=file_size,
            storage_path=storage_path,
            processing_status=DocumentProcessingStatus.UPLOADED.value,
            extracted_content=[],
            extraction_metadata={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        document.file_type = file_type
        document.mime_type = mime_type
        document.file_size = file_size
        document.storage_path = storage_path
        document.processing_status = DocumentProcessingStatus.UPLOADED.value
        document.versions.append(version)
        return document, version

    def set_status(
        self,
        document_id: UUID,
        version_id: UUID,
        status: DocumentProcessingStatus,
        processed: ProcessedDocument | None = None,
        chunks: list[DocumentChunk] | None = None,
    ) -> tuple[Document, DocumentVersion]:
        document = self.documents[document_id]
        version = next(version for version in document.versions if version.id == version_id)
        document.processing_status = status.value
        version.processing_status = status.value
        if processed is not None:
            version.extracted_content = [
                {"sequence": segment.sequence, "text": segment.text, "page_number": segment.page_number,
                 "paragraph_index": segment.paragraph_index, "paragraph_style": segment.paragraph_style,
                 "line_number": segment.line_number, "table_index": segment.table_index,
                 "table_row_index": segment.table_row_index, "table_cell_index": segment.table_cell_index}
                for segment in processed.segments
            ]
            version.extraction_metadata = processed.extraction_metadata
        if chunks is not None:
            self.chunks[version_id] = chunks
        return document, version

    def soft_delete(self, document: Document) -> None:
        document.deleted_at = datetime.now(UTC)

    def delete(self, document: Document) -> None:
        self.documents.pop(document.id)

    def has_scoped_conversation(self, document_id: UUID) -> bool:
        return document_id in self.scoped_documents

    def rollback(self) -> None:
        return None


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_upload = False
        self.fail_delete = False

    def upload(self, path: str, content: bytes, content_type: str) -> None:
        if self.fail_upload:
            raise StorageError("storage unavailable")
        self.objects[path] = content

    def delete(self, path: str) -> None:
        if self.fail_delete:
            raise StorageError("storage unavailable")
        self.objects.pop(path, None)


@pytest.fixture
def persistence() -> tuple[DocumentLibraryService, MemoryRepository, MemoryStorage]:
    repository = MemoryRepository()
    storage = MemoryStorage()
    service = DocumentLibraryService(
        repository=repository,
        storage=storage,
        processor=DocumentProcessingService(max_file_size_bytes=1_000_000),
    )
    return service, repository, storage


def upload(service: DocumentLibraryService, content: bytes = b"hello\nworld"):
    return service.upload(WORKSPACE_ID, str(USER_ID), "notes.txt", "text/plain", content)


def test_document_creation_persists_file_version_and_extraction(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, repository, storage = persistence

    document = upload(service)

    stored = repository.documents[document.id]
    version = stored.versions[0]
    assert document.processing_status == "ready"
    assert document.current_version == 1
    assert version.extracted_content[0]["line_number"] == 1
    assert version.extraction_metadata["segment_count"] == 2
    assert repository.chunks[version.id]
    assert storage.objects[version.storage_path] == b"hello\nworld"


def test_document_retrieval_returns_versions_and_extracted_metadata(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, _, _ = persistence
    created = upload(service)

    retrieved = service.get_document(created.id, str(USER_ID))

    assert retrieved.id == created.id
    assert retrieved.versions[0].extracted_content[1].line_number == 2


def test_document_listing_is_workspace_scoped(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, _, _ = persistence
    upload(service)

    listed = service.list_documents(WORKSPACE_ID, str(USER_ID))

    assert [document.filename for document in listed] == ["notes.txt"]
    with pytest.raises(AppError) as error:
        service.list_documents(OTHER_WORKSPACE_ID, str(USER_ID))
    assert error.value.status_code == 404


def test_document_deletion_removes_object_and_database_record(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, repository, storage = persistence
    created = upload(service)
    path = next(iter(storage.objects))

    service.delete_document(created.id, str(USER_ID))

    assert created.id not in repository.documents
    assert path not in storage.objects


def test_document_scope_conflict_is_checked_before_storage_deletion(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, repository, storage = persistence
    created = upload(service)
    repository.scoped_documents.add(created.id)
    saved_objects = dict(storage.objects)

    with pytest.raises(AppError) as error:
        service.delete_document(created.id, str(USER_ID))

    assert error.value.status_code == 409
    assert error.value.code == "document_conversation_scope_conflict"
    assert storage.objects == saved_objects
    assert created.id in repository.documents


def test_workspace_isolation_rejects_upload_before_storage_write(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, _, storage = persistence

    with pytest.raises(AppError) as error:
        service.upload(OTHER_WORKSPACE_ID, str(USER_ID), "notes.txt", "text/plain", b"hello")

    assert error.value.status_code == 404
    assert storage.objects == {}


def test_unauthorized_user_cannot_read_or_delete_document(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, _, _ = persistence
    created = upload(service)
    other_user_id = str(uuid4())

    with pytest.raises(AppError) as get_error:
        service.get_document(created.id, other_user_id)
    with pytest.raises(AppError) as delete_error:
        service.delete_document(created.id, other_user_id)

    assert get_error.value.status_code == delete_error.value.status_code == 404


def test_same_content_is_idempotent_and_changed_content_creates_new_version(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, repository, _ = persistence
    first = upload(service)
    duplicate = upload(service)
    updated = upload(service, b"changed content")

    assert duplicate.id == first.id
    assert updated.id == first.id
    assert updated.current_version == 2
    assert len(repository.documents[first.id].versions) == 2


def test_persistence_failure_does_not_write_to_storage(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, repository, storage = persistence
    repository.fail_start = True

    with pytest.raises(AppError) as error:
        upload(service)

    assert error.value.code == "persistence_unavailable"
    assert storage.objects == {}


def test_storage_failure_marks_document_failed(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, repository, storage = persistence
    storage.fail_upload = True

    with pytest.raises(AppError) as error:
        upload(service)

    document = next(iter(repository.documents.values()))
    assert error.value.status_code == 503
    assert document.processing_status == "failed"


def test_upload_endpoint_uses_overridden_persistence_dependencies(
    persistence: tuple[DocumentLibraryService, MemoryRepository, MemoryStorage],
) -> None:
    service, _, _ = persistence
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(id=str(USER_ID))
    app.dependency_overrides[get_document_library_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/workspaces/{WORKSPACE_ID}/documents",
                files={"file": ("notes.txt", b"route upload", "text/plain")},
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_document_library_service, None)

    assert response.status_code == 201
    assert response.json()["processing_status"] == "ready"
    assert response.json()["current_version"] == 1