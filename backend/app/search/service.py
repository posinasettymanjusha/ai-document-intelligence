from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import AppError
from app.documents.repository import DocumentRepository
from app.embeddings.service import EmbeddingService
from app.search.schemas import SemanticSearchRequest, SemanticSearchResult


class SemanticSearchService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self._repository = repository
        self._embedding_service = embedding_service

    def search(
        self,
        workspace_id: UUID,
        user_id: str,
        request: SemanticSearchRequest,
    ) -> list[SemanticSearchResult]:
        try:
            user_uuid = UUID(user_id)
        except ValueError as error:
            raise AppError(401, "unauthorized", "A valid user identity is required.") from error

        try:
            allowed = self._repository.is_workspace_member(workspace_id, user_uuid)
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "Workspace access could not be verified.",
            ) from error
        if not allowed:
            raise AppError(404, "not_found", "The requested resource was not found.")

        try:
            query_embedding = self._embedding_service.embed_text(request.query)
        except Exception as error:
            raise AppError(
                503,
                "embedding_unavailable",
                "The search query could not be embedded.",
            ) from error

        try:
            matches = self._repository.search_chunks(
                workspace_id=workspace_id,
                user_id=user_uuid,
                query_vector=query_embedding.vector,
                embedding_model=query_embedding.model_id,
                top_k=request.top_k,
                document_id=request.document_id,
                version_id=request.version_id,
            )
        except SQLAlchemyError as error:
            self._repository.rollback()
            raise AppError(
                503,
                "persistence_unavailable",
                "Search results could not be loaded.",
            ) from error

        results: list[SemanticSearchResult] = []
        for chunk, document, version, distance in matches:
            cosine_distance = float(distance)
            results.append(
                SemanticSearchResult(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    filename=document.filename,
                    version_id=version.id,
                    version_number=version.version_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    cosine_distance=cosine_distance,
                    similarity=1.0 - cosine_distance,
                    source_metadata=chunk.source_metadata,
                )
            )
        return results