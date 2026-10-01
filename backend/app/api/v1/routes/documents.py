from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.chunking.service import DocumentChunkingService
from app.core.config import settings
from app.db.session import get_db
from app.embeddings.factory import create_embedding_service
from app.embeddings.generation import EmbeddingGenerationService
from app.embeddings.schemas import EmbeddingGenerationRequest, EmbeddingGenerationResponse
from app.documents.library import DocumentLibraryService
from app.documents.repository import DocumentRepository
from app.documents.schemas import DocumentListItemResponse, DocumentResponse
from app.documents.service import DocumentProcessingService
from app.integrations.storage import ObjectStorage, SupabaseStorage

router = APIRouter()


def get_document_repository(
    session: Annotated[Session, Depends(get_db)],
) -> DocumentRepository:
    return DocumentRepository(session)


def get_document_storage() -> ObjectStorage:
    return SupabaseStorage()


def get_document_library_service(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    storage: Annotated[ObjectStorage, Depends(get_document_storage)],
) -> DocumentLibraryService:
    return DocumentLibraryService(
        repository=repository,
        storage=storage,
        processor=DocumentProcessingService(settings.max_document_size_bytes),
        chunker=DocumentChunkingService(
            chunk_size=settings.document_chunk_size,
            chunk_overlap=settings.document_chunk_overlap,
            minimum_chunk_size=settings.document_minimum_chunk_size,
        ),
    )


def get_embedding_generation_service(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> EmbeddingGenerationService:
    return EmbeddingGenerationService(
        repository=repository,
        embedding_service=create_embedding_service(),
        batch_size=settings.embedding_batch_size,
    )


@router.post(
    "/workspaces/{workspace_id}/documents",
    response_model=DocumentResponse,
    status_code=201,
    dependencies=[Depends(get_current_user)],
    summary="Upload and persist a document in a workspace",
)
def create_document(
    workspace_id: UUID,
    file: Annotated[UploadFile, File(...)],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[DocumentLibraryService, Depends(get_document_library_service)],
) -> DocumentResponse:
    try:
        content = file.file.read(settings.max_document_size_bytes + 1)
    finally:
        file.file.close()
    return service.upload(workspace_id, user.id, file.filename or "", file.content_type, content)


@router.get(
    "/workspaces/{workspace_id}/documents",
    response_model=list[DocumentListItemResponse],
    dependencies=[Depends(get_current_user)],
    summary="List documents in a workspace",
)
def list_documents(
    workspace_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[DocumentLibraryService, Depends(get_document_library_service)],
) -> list[DocumentListItemResponse]:
    return service.list_documents(workspace_id, user.id)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    dependencies=[Depends(get_current_user)],
    summary="Get a document and its persisted versions",
)
def get_document(
    document_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[DocumentLibraryService, Depends(get_document_library_service)],
) -> DocumentResponse:
    return service.get_document(document_id, user.id)


@router.delete(
    "/documents/{document_id}",
    status_code=204,
    dependencies=[Depends(get_current_user)],
    summary="Delete a document and its private file versions",
)
def delete_document(
    document_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[DocumentLibraryService, Depends(get_document_library_service)],
) -> None:
    service.delete_document(document_id, user.id)


@router.post(
    "/documents/{document_id}/versions/{version_id}/embeddings",
    response_model=EmbeddingGenerationResponse,
    dependencies=[Depends(get_current_user)],
    summary="Explicitly generate embeddings for pending document chunks",
)
def generate_document_embeddings(
    document_id: UUID,
    version_id: UUID,
    request: Annotated[EmbeddingGenerationRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[EmbeddingGenerationService, Depends(get_embedding_generation_service)],
) -> EmbeddingGenerationResponse:
    summary = service.generate_for_version(
        document_id,
        version_id,
        user.id,
        retry_failed=request.retry_failed,
    )
    return EmbeddingGenerationResponse(
        attempted_chunks=summary.attempted_chunks,
        ready_chunks=summary.ready_chunks,
        failed_chunks=summary.failed_chunks,
        failure_message=summary.failure_message,
    )