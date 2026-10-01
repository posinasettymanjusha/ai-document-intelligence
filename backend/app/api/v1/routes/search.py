from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends

from app.api.v1.routes.documents import get_document_repository
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.documents.repository import DocumentRepository
from app.embeddings.factory import create_embedding_service
from app.search.schemas import SemanticSearchRequest, SemanticSearchResult
from app.search.service import SemanticSearchService

router = APIRouter()


def get_semantic_search_service(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> SemanticSearchService:
    return SemanticSearchService(
        repository=repository,
        embedding_service=create_embedding_service(),
    )


@router.post(
    "/workspaces/{workspace_id}/search",
    response_model=list[SemanticSearchResult],
    dependencies=[Depends(get_current_user)],
    summary="Search ready document chunks in a workspace",
)
def search_workspace(
    workspace_id: UUID,
    request: Annotated[SemanticSearchRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[SemanticSearchService, Depends(get_semantic_search_service)],
) -> list[SemanticSearchResult]:
    return service.search(workspace_id, user.id, request)