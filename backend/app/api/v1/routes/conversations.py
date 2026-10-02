from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.api.v1.routes.answers import get_grounded_answer_service
from app.api.v1.routes.documents import get_document_repository
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthenticatedUser
from app.conversations.repository import ConversationRepository
from app.conversations.schemas import (
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationMessagePairResponse,
    ConversationMessageRequest,
    ConversationResponse,
)
from app.conversations.service import ConversationService
from app.core.errors import AppError
from app.db.session import get_db
from app.documents.repository import DocumentRepository
from app.rag.service import GroundedAnswerService

router = APIRouter()


def get_conversation_repository(
    session: Annotated[Session, Depends(get_db)],
    document_repository: Annotated[
        DocumentRepository,
        Depends(get_document_repository),
    ],
) -> ConversationRepository:
    return ConversationRepository(session, document_repository)


def get_conversation_service(
    repository: Annotated[
        ConversationRepository,
        Depends(get_conversation_repository),
    ],
    grounded_answer_service: Annotated[
        GroundedAnswerService,
        Depends(get_grounded_answer_service),
    ],
) -> ConversationService:
    return ConversationService(repository, grounded_answer_service)


@router.post(
    "/workspaces/{workspace_id}/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(get_current_user)],
    summary="Create a private workspace conversation",
)
def create_conversation(
    workspace_id: UUID,
    request: Annotated[ConversationCreateRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ConversationResponse:
    return service.create(workspace_id, user.id, request)


@router.get(
    "/workspaces/{workspace_id}/conversations",
    response_model=ConversationListResponse,
    dependencies=[Depends(get_current_user)],
    summary="List the current user's workspace conversations",
)
def list_conversations(
    workspace_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[ConversationService, Depends(get_conversation_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    before_updated_at: Annotated[datetime | None, Query()] = None,
    before_id: UUID | None = None,
) -> ConversationListResponse:
    if (before_updated_at is None) != (before_id is None):
        raise AppError(422, "invalid_cursor", "Both conversation cursor values are required.")
    return service.list_conversations(
        workspace_id,
        user.id,
        limit=limit,
        before_updated_at=before_updated_at,
        before_id=before_id,
    )


@router.get(
    "/workspaces/{workspace_id}/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    dependencies=[Depends(get_current_user)],
    summary="Get a conversation and an ordered message page",
)
def get_conversation(
    workspace_id: UUID,
    conversation_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[ConversationService, Depends(get_conversation_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    before_sequence: Annotated[int | None, Query(ge=1)] = None,
) -> ConversationDetailResponse:
    return service.get(
        workspace_id,
        conversation_id,
        user.id,
        message_limit=limit,
        before_sequence=before_sequence,
    )


@router.post(
    "/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
    response_model=ConversationMessagePairResponse,
    dependencies=[Depends(get_current_user)],
    summary="Submit a question in a document conversation",
)
def send_conversation_message(
    workspace_id: UUID,
    conversation_id: UUID,
    request: Annotated[ConversationMessageRequest, Body()],
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ConversationMessagePairResponse:
    return service.send_message(workspace_id, conversation_id, user.id, request)


@router.delete(
    "/workspaces/{workspace_id}/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(get_current_user)],
    summary="Delete a private conversation and its messages",
)
def delete_conversation(
    workspace_id: UUID,
    conversation_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> Response:
    service.delete(workspace_id, conversation_id, user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)