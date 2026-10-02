from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends

from app.api.v1.routes.answers import get_gemini_generation_provider
from app.api.v1.routes.documents import get_document_repository
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.document_intelligence.evidence import DocumentEvidenceService
from app.document_intelligence.schemas import (
    DocumentComparisonRequest,
    DocumentComparisonResponse,
    DocumentOverviewRequest,
    DocumentOverviewResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
)
from app.document_intelligence.service import DocumentIntelligenceService
from app.documents.repository import DocumentRepository
from app.integrations.gemini_generation import GeminiGenerationProvider

router = APIRouter()


def get_document_intelligence_service(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    generation_provider: Annotated[
        GeminiGenerationProvider,
        Depends(get_gemini_generation_provider),
    ],
) -> DocumentIntelligenceService:
    return DocumentIntelligenceService(
        DocumentEvidenceService(repository),
        generation_provider,
    )


@router.post(
    "/workspaces/{workspace_id}/documents/{document_id}/versions/{version_id}/summary",
    response_model=DocumentSummaryResponse,
    dependencies=[Depends(get_current_user)],
    summary="Generate a grounded summary of a document version",
)
def summarize_document_version(
    workspace_id: UUID,
    document_id: UUID,
    version_id: UUID,
    request: Annotated[DocumentSummaryRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[DocumentIntelligenceService, Depends(get_document_intelligence_service)],
) -> DocumentSummaryResponse:
    return service.summarize(
        workspace_id,
        document_id,
        version_id,
        user.id,
        request,
    )


@router.post(
    "/workspaces/{workspace_id}/documents/{document_id}/versions/{version_id}/key-information",
    response_model=DocumentOverviewResponse,
    dependencies=[Depends(get_current_user)],
    summary="Extract the fixed document overview profile from a document version",
)
def extract_document_overview(
    workspace_id: UUID,
    document_id: UUID,
    version_id: UUID,
    request: Annotated[DocumentOverviewRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[DocumentIntelligenceService, Depends(get_document_intelligence_service)],
) -> DocumentOverviewResponse:
    return service.extract_key_information(
        workspace_id,
        document_id,
        version_id,
        user.id,
    )


@router.post(
    "/workspaces/{workspace_id}/document-comparisons",
    response_model=DocumentComparisonResponse,
    dependencies=[Depends(get_current_user)],
    summary="Compare two authorized document versions with grounded evidence",
)
def compare_documents(
    workspace_id: UUID,
    request: Annotated[DocumentComparisonRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[DocumentIntelligenceService, Depends(get_document_intelligence_service)],
) -> DocumentComparisonResponse:
    return service.compare(workspace_id, user.id, request)